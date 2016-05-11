{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module Main where
import ClassyPrelude
import qualified NLP.Albemarle.Tokens as Tokens

main :: IO ()
main = do
  print $ Tokens.wordTokenize "this example"
