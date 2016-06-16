{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
-- | Albemarle, natural language processing for Haskell
module NLP.Albemarle (
  BagOfWords
) where
import ClassyPrelude
import qualified Data.Vector.Unboxed as Vec

-- | Vector of sorted (word ID, count)
type BagOfWords = Vec.Vector (Int, Int)
