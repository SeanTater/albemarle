{-# LANGUAGE OverloadedStrings, BangPatterns #-}
module NLP.Albemarle.Test.LanguageModelEntropy where
import Prelude hiding (words)
import Test.Hspec
import Data.Monoid
import Data.Foldable
import Data.Ord
import Data.List hiding (words)
import Data.Text (Text, words)
import qualified Data.HashMap.Strict as HashMap
import NLP.Albemarle.Test.Fixture as Fixture
import NLP.Albemarle.Dict as Dict
import Lens.Micro
import Text.Printf

test = describe "Language model predictive quality" $ let
    boolAsInt x = if x then 1 else 0
    foldMap' f = foldl' (\m -> mappend m . f) mempty
    logprob p = - log p / log 2
    fi i = fromIntegral i
  in do
    it "Using only the most likely word" $ do
      fixture <- Fixture.generate
      -- All hists have at least '', never empty.
      -- maximum is partial but not an issue here
      let h = fixture^.dict.counts
          top = maximum $ snd <$> HashMap.toList h
          population = sum $ snd <$> HashMap.toList h
      printf "Always choose only most common word: %.02f%% accuracy (no bits)\n"
        $ (100 * (fi top / fi population) :: Double)
      True `shouldBe` True

    it "Using prior probabilities" $ do
      fixture <- Fixture.generate
      -- All hists have at least '', never empty
      let h = fixture^.dict.counts
          total_freq = sum $ map snd $ HashMap.toList h
          freq w = HashMap.lookupDefault 0 w h
          prob w = fi (freq w) / fi total_freq :: Double
          (bits, corpus_len) = over both getSum
            $ foldMap' (\x -> (Sum $ logprob $ prob x, Sum 1))
            $ concat $ fixture^.docs
      printf "Priors: %.02f bits/word\n" $ bits / corpus_len
      True `shouldBe` True
