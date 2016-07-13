{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, BangPatterns #-}
import ClassyPrelude hiding (head, tail)
import Prelude (head, tail)
import Test.Hspec
import Test.QuickCheck
import qualified Criterion.Main as Criterion
import Lens.Micro

import Control.Exception (evaluate)
import qualified Data.Vector.Storable as SVec
import qualified Data.Vector.Generic as Vec
import qualified Data.Binary as Bin

import NLP.Albemarle
import qualified NLP.Albemarle.Tokens as Tokens
import NLP.Albemarle.Dict (Dict, counts, ids, hist)
import qualified NLP.Albemarle.Dict as Dict
import NLP.Albemarle.LSA (termvectors, topicweights)
import qualified NLP.Albemarle.LSA as LSA
import qualified NLP.Albemarle.Test.Dict
import qualified NLP.Albemarle.Test.GloVe
import qualified NLP.Albemarle.Examples.Webpage
import System.CPUTime
import Text.Printf
import Data.Text.Encoding.Error (lenientDecode)
import System.Directory (doesFileExist)

import qualified Data.Text as Text
import qualified Data.HashSet as HashSet
import qualified Data.HashMap.Strict as HashMap
import System.IO.Streams (InputStream, OutputStream)
import qualified System.IO.Streams as Streams
import Numeric.LinearAlgebra (Matrix)
import qualified Numeric.LinearAlgebra as HMatrix

main :: IO ()
main = hspec $ do
  NLP.Albemarle.Examples.Webpage.test
  NLP.Albemarle.Test.Dict.test
  NLP.Albemarle.Test.GloVe.test

  let sentences = words <$> [
        "Maybe not today. Maybe not tomorrow. But soon.",
        "Pay no attention to the man behind the curtain.",
        "Life is like a box of chocolates."]:: [[Text]]

  describe "Streaming (semi-Gensim) Style Topic Analysis" $ do
    let mean :: SVec.Vector Double -> Double
        mean x = Vec.sum x / fromIntegral (Vec.length x)
        -- Root mean squared error
        rmse l r = mean (Vec.zipWith (\x y -> (x-y)**2) (flat l) (flat r)) ** 0.5
        -- Mean signal to noise ratio
        -- - we tank the PSNR because most things are really close to 0
        -- - in fact we avoid handling 0's altogether
        msnr l r = mean $ Vec.zipWith (\x y -> abs $ x-y / max 1 (min x y)) (flat l) (flat r)
        flat = HMatrix.flatten
        mult u s vt = mconcat [u, HMatrix.diag s, HMatrix.tr vt] :: HMatrix.Matrix Double

    it "performs sparse stochastic truncated SVD with SVDLIBC" $ do
      let termdoc = "termdoc.small.nonzero.txt"
      [zippedfile, originalfile] <- sequence $ doesFileExist <$> [termdoc++".gz", termdoc]
      when ( zippedfile && not originalfile ) $
        Streams.withFileAsInput "termdoc.small.nonzero.txt.gz" $ \infl ->
          Streams.withFileAsOutput "termdoc.small.nonzero.txt" $ \outfl ->
            Streams.gunzip infl >>= Streams.connectTo outfl
      matrix <- HMatrix.loadMatrix "termdoc.small.nonzero.txt"
      let (u, sigma, vt) = LSA.batchLSA 100 $ LSA.sparsify matrix
      -- typically -> RMSE: 0.06464712163523892
      -- printf "RMSE: %f\n" $ rmse (mconcat [u, HMatrix.diag sigma, HMatrix.tr vt]) matrix
      rmse (mult u sigma vt) matrix < 0.12 `shouldBe` True
      -- typically -> MSNR: 0.010548326765305554
      -- printf "MSNR: %f\n" $ msnr (mconcat [u, HMatrix.diag sigma, HMatrix.tr vt]) matrix
      msnr (mult u sigma vt) matrix < 0.02 `shouldBe` True

    it "performs topic analysis starting from text" $ do
      (final_dict, model) <- Streams.withFileAsInput "kjv.verses.txt.gz" $ \file ->
        Streams.gunzip file
        >>= Streams.lines
        >>= Streams.decodeUtf8With lenientDecode
        >>= Streams.map Tokens.wordTokenize
        >>= Streams.chunkList 373 -- Yep, I'm making this stuff up.
        >>= Streams.map (\chunk -> let
          dict = Dict.dictifyAllWords chunk
          sparsem = Dict.asSparseMatrix dict chunk
          lsa = LSA.lsa 222 sparsem -- Too many topics. Should not cause errors
          in (dict, lsa))
        >>= Streams.fold (\(!d1, !lsa1) (d2, lsa2) -> let
          d3 = Dict.filterDict 2 0.5 100 $ d1 <> d2 -- Not nearly enough words
          lsa3 = LSA.rebase d1 d3 lsa1 <> LSA.rebase d2 d3 lsa2
          in (d3, lsa3)) (mempty, mempty)

      -- It should use all 100 words allowed plus the unknown
      HashMap.size (final_dict^.counts.hist) `shouldBe` 101
      -- It should have a full size LSA as well
      HMatrix.size (model^.termvectors) `shouldBe` (222, 101)
      -- Plus topic weights
      HMatrix.size (model^.topicweights) `shouldBe` 222

  describe "Monoid style Topic Analysis" $
    it "creates LSA Models" $ do
      let dict = Dict.dictifyAllWords sentences
      let lsavecs = LSA.lsa 2
            $ Dict.asSparseMatrix dict sentences
      -- Sadly, I can't think of much to assert in this test so let's use size
      -- There are 21 unique words plus one unknown
      HMatrix.size (lsavecs^.termvectors) `shouldBe` (2,22)
      -- It should use both topics
      HMatrix.size (lsavecs^.topicweights) `shouldBe` 2

-- Thanks to Ying He for the following example text and proper
-- tokenizations.
--weightGain :: Text
--weightGain = unwords [
--  "Independent of current body composition, IGF-I levels at 5 yr were ",
--  "significantly associated with rate of weight gain between 0-2 yr",
--  "(beta = 0.19; P < 0.0005), and children who showed postnatal",
--  "catch-up growth (i.e. those who showed gains in weight or length",
--  "between 0-2 yr by >0.67 SD score) had higher IGF-I levels than other",
--  "children (P = 0.02)."]

--weightGainTokens :: [Text]
--weightGainTokens = words $ unwords [
--  "Independent of current body composition , IGF - I levels at 5 yr were",
--  "significantly associated with rate of weight gain between 0 - 2 yr ( beta",
--  "= 0.19 ; P < 0.0005 ) , and children who showed postnatal catch - up",
--  "growth ( i.e . those who showed gains in weight or length between 0 - 2 yr",
--  "by > 0.67 SD score ) had higher IGF - I levels than other children",
--  "( P = 0.02 ) ."] -- i.e. is wrong, but we're calling this close enough.
