-- | Albemarle, natural language processing for Haskell
module NLP.Albemarle (
  Document
) where
import ClassyPrelude
import qualified Data.Vector.Unboxed as Vec

-- | Vector of sorted (word ID, count)
type Document = Vec.Vector (Int, Int)
