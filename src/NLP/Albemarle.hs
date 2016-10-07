{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
{-# LANGUAGE FlexibleContexts, UndecidableInstances #-} -- required for HMatrix
-- | Albemarle, natural language processing for Haskell
module NLP.Albemarle (
  SparseMatrix(..),
  SparseVector(..),
  DenseVector(..)
) where
import ClassyPrelude hiding (Vector)
import Numeric.LinearAlgebra
import qualified Data.Vector.Unboxed as Vec

-- | Vector of sorted (word ID, count)
data SparseVector = SparseVector Int [(Int, Double)]
  deriving (Show, Eq)
data SparseMatrix = SparseMatrix Int [SparseVector]
  deriving (Show, Eq)
type DenseVector = Vector Double

instance (Container Vector t, Eq t, Num (Vector t), Product t) => Semigroup (Matrix t) where
  (<>) = mappend
