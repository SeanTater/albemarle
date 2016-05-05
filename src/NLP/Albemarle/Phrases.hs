{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module NLP.Albemarle.Phrases
    ( discover
    , use
    ) where
import ClassyPrelude.Conduit
import Data.Text (Text)
import qualified Data.Text as Text
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap
import qualified NLP.Albemarle.Dictionary as Dictionary

-- | Tokenize text according to the rule of leftmost longest from a set of
--   permitted tokens.
--
--   This uses a HashMap (There may be room for improvement using Tries)
--   It will only produce elements of the dictionary (as opposed to extendModel)
useOne :: HashMap Text Int -> [Text] -> [Text]
useOne dictionary [] = []
useOne dictionary (first:rest) =
  useModel first rest
  where
    useModel target [] = [target | HashMap.member target dictionary]
    useModel target tokens@(next:rest) =
      if HashMap.member next_target dictionary
        then useModel next_target rest
        else target : useModel next rest
      where
        next_target = target ++ "_" ++ next

use :: HashMap Text Int -> Conduit [Text] IO [Text]
use dictionary = mapC $ useOne dictionary

-- | Tokenize text according to the rule of leftmost longest from a set of
--   permitted tokens.
--
--   This uses a HashMap (There may be room for improvement using Tries)
--   It will only produce elements of the dictionary, plus one extra token.
--   (as opposed to applyModel).
extendOne :: HashMap Text Int -> [Text] -> [Text]
extendOne dictionary [] = []
extendOne dictionary (first:rest) =
  extendModel first rest
  where
    extendModel target [] = [target | HashMap.member target dictionary]
    extendModel target tokens@(next:rest) =
      if HashMap.member next_target dictionary
        then extendModel next_target rest
        else next_target : extendModel next rest
      where
        next_target = target ++ "_" ++ next

-- | Find phrases in text according to their frequency relative to those of ins
--   constituents
extend :: HashMap Text Int -> Conduit [Text] IO [Text]
extend dictionary = mapC $ discoverOne dictionary

discover :: HashMap Text Int -> HashMap Text Int -> HashMap Text Int
discover previous_dict next_dict =
