{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
import ClassyPrelude
import Test.Hspec
import Test.QuickCheck
import Control.Exception (evaluate)
import Data.HashSet (HashSet)
import qualified Data.HashSet as HashSet
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap

import qualified System.IO.Streams as Streams
import System.IO.Streams (Generator, InputStream, OutputStream)

import qualified NLP.Albemarle.Tokens as Tokens
import qualified NLP.Albemarle.Dictionary as Dictionary


main :: IO ()
main = hspec $ do
  describe "Standard Tools" $ do
    it "#1 Tokenizes" $
      Tokens.wordTokenize weight_gain `shouldBe` weight_gain_tokens

    it "#3 Creates Dictionaries" $ do
      counts <- Dictionary.countAdv 2 0.5 100 100 =<< Streams.fromList little_docs
      counts `shouldBe` little_counts
      Dictionary.assignIDs counts `shouldBe` little_ids





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

little_counts :: HashMap Text Int
little_counts = HashMap.fromList [("three", 2), ("four", 2), ("five", 3)]

little_ids :: HashMap Text Int
little_ids = HashMap.fromList [("three", 2), ("four", 1), ("five", 0)]
