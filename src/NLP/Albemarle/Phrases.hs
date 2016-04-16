{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module NLP.Albemarle.Phrases
    ( applyModel
    , extendModel
    , generateModel
    ) where
import ClassyPrelude.Conduit
import Data.Text (Text)
import qualified Data.Text as Text
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap
import qualified NLP.Albemarle.Dictionary

-- ! Tokenize text according to the rule of leftmost longest from a set of
-- ! permitted tokens.
-- ! This uses a HashMap (There may be room for improvement using Tries)
-- ! It will only produce elements of the dictionary (as opposed to extendModel)
applyModel :: HashMap Text Int -> Text -> [Text] -> [Text]
applyModel dictionary target [] = [target | HashMap.member target dictionary]
applyModel dictionary target tokens@(next:rest) =
  if HashMap.member next_target dictionary
    then applyModel dictionary next_target rest
    else target : applyModel dictionary next rest
  where
    next_target = target ++ "_" ++ next

-- ! Tokenize text according to the rule of leftmost longest from a set of
-- ! permitted tokens.
-- ! This uses a HashMap (There may be room for improvement using Tries)
-- ! It will only produce elements of the dictionary, plus one extra token.
-- ! (as opposed to applyModel).
extendModel :: HashMap Text Int -> Text -> [Text] -> [Text]
extendModel dictionary target [] = [target | HashMap.member target dictionary]
extendModel dictionary target tokens@(next:rest) =
  if HashMap.member next_target dictionary
    then extendModel dictionary next_target rest
    else next_target : extendModel dictionary next rest
  where
    next_target = target ++ "_" ++ next

-- ! Tokenize text according to the rule of leftmost longest from a set of
-- ! permitted tokens.
-- ! This uses a HashMap (There may be room for improvement using Tries)
-- ! It will only produce elements of the dictionary, plus one extra token.
-- ! (as opposed to applyModel).
generateModel :: [Text] -> HashMap Text Int
generateModel tokens =
  Dictionary
