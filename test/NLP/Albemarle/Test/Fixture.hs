{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
module NLP.Albemarle.Test.Fixture where
import Prelude hiding (words, lines, readFile)
import Test.Hspec
import Data.List hiding (words, lines)
import Data.Text (Text, words, lines)
import Data.Text.IO (readFile)
import qualified NLP.Albemarle.Tokens as Tokens
import qualified NLP.Albemarle.Dict as Dict
import Lens.Micro.TH

data Fixture = Fixture {
  _docs :: [[Text]],
  _dict :: Dict.Dict
}
makeLenses ''Fixture

generate :: IO Fixture
generate = do
  docs <- (fmap.fmap) Tokens.wordTokenize
    $ fmap lines
    $ readFile "kjv.lines.txt"
  return $ Fixture {
    _docs = docs,
    _dict = Dict.dictifyAllWords docs
  }
