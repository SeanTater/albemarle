{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
{-# LANGUAGE FlexibleContexts, UndecidableInstances #-} -- required for HMatrix
-- | Albemarle, natural language processing for Haskell
module NLP.Albemarle (
  BagOfWords
) where
import ClassyPrelude hiding (Vector)
import Numeric.LinearAlgebra
import qualified Data.Vector.Unboxed as Vec

-- | Vector of sorted (word ID, count)
type BagOfWords = Vec.Vector (Int, Int)

instance (Container Vector t, Eq t, Num (Vector t), Product t) => Semigroup (Matrix t) where
  (<>) = mappend
