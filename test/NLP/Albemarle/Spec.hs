{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
import ClassyPrelude hiding (head, tail)
import Prelude (head, tail)
import Test.Hspec
import Test.QuickCheck
import qualified Criterion.Main as Criterion

import Control.Exception (evaluate)
import qualified Data.Vector.Storable as SVec
import qualified Data.Vector.Generic as Vec
import qualified Data.Binary as Bin

import NLP.Albemarle
import qualified NLP.Albemarle.Tokens as Tokens
import qualified NLP.Albemarle.Dictionary as Dictionary
import qualified NLP.Albemarle.Dict as Dict
import qualified NLP.Albemarle.LSA as LSA
import qualified NLP.Albemarle.Sparse as Sparse
import qualified NLP.Albemarle.Examples.Webpage as Webpage
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
  Webpage.test
  describe "Gensim-style Dictionary" $ do
    it "#1 Tokenizes" $
      Tokens.wordTokenize weight_gain `shouldBe` weight_gain_tokens

    it "#3 Creates Dictionaries" $ do
      counts <- Streams.fromList little_docs
          >>= Dictionary.countAdv 2 0.5 100 100
      counts `shouldBe` little_counts
      Dictionary.assignIDs counts `shouldBe` little_ids

    it "#4 Applies Dictionaries" $ do
      dict <- Streams.fromList little_docs
          >>= Dictionary.discoverAdv 2 0.5 100 100
      ids <- Streams.fromList little_docs
          >>= Dictionary.apply dict
          >>= Streams.toList
      ids `shouldBe` little_sparse_vectors

  describe "Monoid-style Dictionary" $ do
    let sentences = words <$> [
          "Maybe not today. Maybe not tomorrow. But soon.",
          "Pay no attention to the man behind the curtain.",
          "Life is like a box of chocolates."]:: [[Text]]

    it "Generates single dictionaries" $ do
      let d = Dict.dictify $ head sentences
      Dict.idOf d "tomorrow." `shouldBe` 6
      Dict.countOf d "tomorrow." `shouldBe` 1
      Dict.idOf d "Maybe" `shouldBe` 2
      Dict.countOf d "Maybe" `shouldBe` 2
      Dict.idOf d "punk?" `shouldBe` 0
      Dict.countOf d "punk?" `shouldBe` 0
    it "Generates all-words dictionaries" $ do
      let d = Dict.dictifyAllWords sentences
      Dict.idOf d "Maybe" `shouldBe` 3
      Dict.countOf d "Maybe" `shouldBe` 2
    it "Generates first-words dictionaries" $ do
      let d = Dict.dictifyFirstWords sentences
      Dict.idOf d "Maybe" `shouldBe` 3
      Dict.countOf d "Maybe" `shouldBe` 1
    it "Merges dictionaries" $ do
      let d1 = Dict.dictify $ head sentences
      let d2 = Dict.dictifyAllWords $ tail sentences
      let d3 = Dict.dictifyAllWords sentences
      d1 <> d2 `shouldBe` d3
      d1 `shouldNotBe` d2 <> d3
    it "Filters dictionaries" $ do
      let d1 = Dict.dictify $ words "Maybe not today . Maybe not tomorrow . But soon ."
      let d2 = Dict.dictify $ words "Maybe not Maybe not"
      -- At least 2 times, not more than 25% (2.75), at most 100 words
      Dict.filterDict 2 0.25 100 d1 `shouldBe` d2
      Dict.idOf d1 "Maybe" `shouldBe` 3

      let letters = [
            "a                                     t     x y  ",
            "a b c d e f g h i j k           q r s t u v x y z",
            "a b c d e f g h i j k l m n o p q r s t u v x y z"]
      let d3 = Dict.dictifyAllWords $ words <$> letters
      let d4 = Dict.filterDict 2 0.25 15 d3
      let uniq_letters = mconcat $ words <$> letters

      for_ uniq_letters $ \letter -> case Dict.countOf d2 letter of
        1 -> Dict.countOf d3 letter `shouldBe` 0 -- delete all
        -- 2 is undefined. It should be uniformly distributed!
        -- So we _don't_ just want r, s, u, v, and z to be deleted.
        3 -> Dict.countOf d3 letter `shouldBe` 3 -- keep all
        otherwise -> True `shouldBe` True
    it "Remaps dictionaries" $ do
      -- The example is the same as in "Filters dictionaries"
      let d1 = Dict.dictify $ words "Maybe not today . Maybe not tomorrow . But soon ."
      let d2 = Dict.dictify $ words "Maybe not Maybe not"
      let remap = Dict.shift d1 d2
      remap (Dict.idOf d1 "Maybe") `shouldBe` (Dict.idOf d2 "Maybe")
      remap (Dict.idOf d1 "") `shouldBe` 0
      remap (Dict.idOf d1 "ggiuyg") `shouldBe` 0
      --                            \, Maybe, not
      Dict.select d1 d2 `shouldBe` [0,    3,    4]

  describe "Topic Analysis" $ do
    let mean :: SVec.Vector Double -> Double
        mean x = Vec.sum x / (fromIntegral $ Vec.length x)
        -- Root mean squared error
        rmse l r = mean (Vec.zipWith (\x y -> (x-y)**2) (flat l) (flat r)) ** 0.5
        -- Mean signal to noise ratio
        -- - we tank the PSNR because most things are really close to 0
        -- - in fact we avoid handling 0's altogether
        msnr l r = mean $ Vec.zipWith (\x y -> abs $ x-y / max 1 (min x y)) (flat l) (flat r)
        flat = HMatrix.flatten
        mult u s vt = mconcat [u, HMatrix.diag s, HMatrix.tr vt] :: HMatrix.Matrix Double

    --it "Performs stochastic truncated SVD" $ do
    --  matrix <- HMatrix.loadMatrix "termdoc.small.txt"
    --  (u, sigma, vt) <- LSA.stochasticTruncatedSVD 50 2 matrix
    --  validate u sigma vt matrix `shouldBe` True

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

    --it "#5 Performs stochastic truncated sparse SVD" $ do
    --  mat_txt <- readFile "termdoc.small.txt"
    --  let mat = fmap (fmap read . words) $ lines mat_txt
    --  let smat = ESP.fromDenseList mat
    --  u <- EigenLSA.eigenLSA 50 2 smat

    --  print $ Eigen.dims u
    --  Eigen.rows u `shouldBe` 1000

    it "performs topic analysis starting from text" $ do
      dict <- Streams.withFileAsInput "kjv.verses.txt.gz" $ \file ->
            Streams.gunzip file
            >>= Streams.lines
            >>= Streams.decodeUtf8With lenientDecode
            >>= Streams.map Tokens.wordTokenize
            >>= Dictionary.discover
      let width = Dictionary.width dict
      csr <- Streams.withFileAsInput "kjv.verses.txt.gz" $ \file ->
            Streams.gunzip file
            >>= Streams.lines
            >>= Streams.decodeUtf8With lenientDecode
            >>= Streams.map Tokens.wordTokenize
            >>= Dictionary.apply dict
            >>= LSA.docsToCSR width
      let (ut, s, v) =  LSA.batchLSA 100 csr
      print $ (HMatrix.size ut, HMatrix.size v)

  --describe "Word2vec" $ do
  --  it "Generates Skip-grams" $ do
  --    False `shouldBe` True



-- Thanks to Ying He for the following example text and proper
-- tokenizations.
weight_gain :: Text
weight_gain = unwords [
  "Independent of current body composition, IGF-I levels at 5 yr were ",
  "significantly associated with rate of weight gain between 0-2 yr",
  "(beta = 0.19; P < 0.0005), and children who showed postnatal",
  "catch-up growth (i.e. those who showed gains in weight or length",
  "between 0-2 yr by >0.67 SD score) had higher IGF-I levels than other",
  "children (P = 0.02)."]

weight_gain_tokens :: [Text]
weight_gain_tokens = words $ unwords [
  "Independent of current body composition , IGF - I levels at 5 yr were",
  "significantly associated with rate of weight gain between 0 - 2 yr ( beta",
  "= 0.19 ; P < 0.0005 ) , and children who showed postnatal catch - up",
  "growth ( i.e . those who showed gains in weight or length between 0 - 2 yr",
  "by > 0.67 SD score ) had higher IGF - I levels than other children",
  "( P = 0.02 ) ."] -- i.e. is wrong, but we're calling this close enough.

little_docs :: [[Text]]
little_docs = words <$> [
  "one two three four five",
  "one two three four five",
  "one two seven eight nine ten",
  "one two five twelve thirteen fourteen",
  "one two fifteen sixteen seventeen",
  "eighteen",
  ""]

little_sparse_vectors :: [BagOfWords]
little_sparse_vectors = Vec.fromList <$> [
  [(0, 2), (1, 1), (2, 1), (3, 1)],
  [(0, 2), (1, 1), (2, 1), (3, 1)],
  [(0, 6)],
  [(0, 5), (1, 1)],
  [(0, 5)],
  [(0, 1)],
  []
  ]

little_counts :: HashMap Text Int
little_counts = HashMap.fromList [("three", 2), ("four", 2), ("five", 3)]

little_ids :: HashMap Text Int
little_ids = HashMap.fromList [("three", 3), ("four", 2), ("five", 1)]
