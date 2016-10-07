{-# LANGUAGE OverloadedStrings #-}
module NLP.Albemarle.Test.Dict (test) where
import Prelude hiding (words)
import Data.Monoid
import Data.Text (Text, words)
import Data.Foldable
import Test.Hspec
import NLP.Albemarle.Dict (Dict, counts, ids)
import qualified NLP.Albemarle.Dict as Dict

test = describe "Monoid-style Dictionary" $ do
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
      _ -> True `shouldBe` True
  it "Remaps dictionaries" $ do
    -- The example is the same as in "Filters dictionaries"
    let d1 = Dict.dictify $ words "Maybe not today . Maybe not tomorrow . But soon ."
    let d2 = Dict.dictify $ words "Maybe not Maybe not"
    let remap = Dict.shift d1 d2
    remap (Dict.idOf d1 "Maybe") `shouldBe` Dict.idOf d2 "Maybe"
    remap (Dict.idOf d1 "") `shouldBe` 0
    remap (Dict.idOf d1 "ggiuyg") `shouldBe` 0
    --                            \, Maybe, not
    Dict.select d1 d2 `shouldBe` [0,    3,    4]
