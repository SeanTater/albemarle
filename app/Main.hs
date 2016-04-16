{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module Main where
import ClassyPrelude.Conduit
import qualified Tokens

main :: IO ()
main = do
  print $ Tokens.wordTokenize "this example"
