{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
import Prelude (read)
import ClassyPrelude
import Test.Hspec
import Test.QuickCheck
import qualified Criterion.Main as Criterion

import Control.Exception (evaluate)
import Data.HashSet (HashSet)
import qualified Data.HashSet as HashSet
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap
import qualified System.IO.Streams as Streams
import System.IO.Streams (Generator, InputStream, OutputStream)
import qualified Data.Vector.Unboxed as Vec
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Data.Binary as Bin
import qualified Data.Eigen.Matrix as Eigen
import qualified Data.Eigen.SparseMatrix as ESP

import NLP.Albemarle
import qualified NLP.Albemarle.Tokens as Tokens
import qualified NLP.Albemarle.Dictionary as Dictionary
import qualified NLP.Albemarle.LSA as LSA
import qualified NLP.Albemarle.Sparse as Sparse
import qualified NLP.Albemarle.Examples.Webpage as Webpage
import qualified NLP.Albemarle.EigenLSA as EigenLSA


main :: IO ()
main = hspec $ do
  Webpage.test
  describe "Standard Tools" $ do
    it "#1 Tokenizes" $
      Tokens.wordTokenize weight_gain `shouldBe` weight_gain_tokens

    it "#3 Creates Dictionaries" $ do
      counts <- Dictionary.countAdv 2 0.5 100 100 =<< Streams.fromList little_docs
      counts `shouldBe` little_counts
      Dictionary.assignIDs counts `shouldBe` little_ids

    it "#4 Applies Dictionaries" $ do
      dict <- Dictionary.discoverAdv 2 0.5 100 100 =<< Streams.fromList little_docs
      ids <- Streams.fromList little_docs >>= Dictionary.apply dict >>= Streams.toList
      ids `shouldBe` little_sparse_vectors

  describe "Topic Analysis" $ do
    it "Performs stochastic truncated SVD" $ do
      matrix <- HMatrix.loadMatrix "termdoc.small.txt"
      (u, sigma, vt) <- LSA.stochasticTruncatedSVD 50 2 matrix

      print $ HMatrix.size u
      print $ HMatrix.size sigma
      print $ HMatrix.size vt

    --it "#5 Performs stochastic truncated sparse SVD" $ do
    --  mat_txt <- readFile "termdoc.small.txt"
    --  let mat = fmap (fmap read . words) $ lines mat_txt
    --  let smat = ESP.fromDenseList mat
    --  u <- EigenLSA.eigenLSA 50 2 smat

    --  print $ Eigen.dims u
    --  Eigen.rows u `shouldBe` 1000

  describe "Word2vec" $ do
    it "Generates Skip-grams" $ do
      False `shouldBe` True



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
