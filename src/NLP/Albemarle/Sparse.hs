{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, BangPatterns, DeriveGeneric #-}
module NLP.Albemarle.Sparse
    ( dense
    , fromDocuments
    , transpose
    , shift
    , size
    , sparse
    , mult
    , multCol
    , SparseMatrix(..)
    , Coord
    , Coords
    ) where
import ClassyPrelude
import NLP.Albemarle
import GHC.Generics (Generic)
import Data.Vector.Binary
import qualified Data.Vector.Unboxed as Vec
import qualified Data.Vector.Storable as SVec
import qualified Data.Vector.Algorithms.Intro as Introsort
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Data.Binary as Bin
import Control.Parallel.Strategies (rpar, rseq, parList, withStrategy, dot, rdeepseq)
type Coord = (Int, Int, Double)
type Coords = Vec.Vector Coord
data SparseMatrix = RowMatrix !Coords | ColMatrix !Coords
  deriving (Show, Eq, Generic)
instance Bin.Binary SparseMatrix

-- | Create a sparse matrix from rows
fromDocuments :: [SparseVector] -> SparseMatrix
fromDocuments rows = RowMatrix $ Vec.concat [ docToSparseRow docid doc | (docid, doc) <- zip [0..] rows ]
  where docToSparseRow docid = Vec.map (\(colid, val) -> (docid, colid, fromIntegral val)) . Vec.fromList

-- | Find the size of a sparse matrix
size :: SparseMatrix -> (Int, Int)
size (RowMatrix mat) = (maxx+1, maxy+1)
  where (maxx, maxy) = Vec.foldr' (\(r, c, _) (mr, mc) -> (max mr r, max mc c)) (-1, -1) mat
size (ColMatrix mat) = (maxx+1, maxy+1)
  where (maxx, maxy) = Vec.foldr' (\(r, c, _) (mr, mc) -> (max mr r, max mc c)) (-1, -1) mat

-- | Convert a sparse matrix into a dense one
dense :: SparseMatrix -> HMatrix.Matrix Double
-- dense (RowMatrix mat) = HMatrix.toDense $ (\(row, col, val) -> ((row, col), val)) <$> Vec.toList mat
-- dense (ColMatrix mat) = HMatrix.toDense $ (\(row, col, val) -> ((row, col), val)) <$> Vec.toList mat
dense (RowMatrix mat) =
  HMatrix.reshape width $ Vec.convert flat_mat
  where
    (width, height) = size $ RowMatrix mat
    flat_mat = Vec.update
      (Vec.replicate (width*height) 0.0) $
      Vec.map (\(r,c,v) -> (r*width+c, v)) mat
dense (ColMatrix mat) = dense $ RowMatrix mat -- a little fib but it works

-- | Convert a dense matrix into a sparse one
sparse :: HMatrix.Matrix Double -> SparseMatrix
sparse mat =
  RowMatrix $ Vec.imap (\ix v -> (
    ix `div` width,
    ix `mod` width,
    v)) $ Vec.convert $ HMatrix.flatten mat
  where (width, height) = HMatrix.size mat
  --RowMatrix $ Vec.fromList [ (ri, ci, val)
  --  | (ri, row) <- zip [0..] lmat,
  --    (ci, val) <- zip [0..] row, abs val > 1e-6 ]
  --where lmat = HMatrix.toLists mat

transpose :: SparseMatrix -> SparseMatrix
transpose (RowMatrix mat) = RowMatrix $ Vec.map (\(row, col, val) -> (col, row, val)) mat
transpose (ColMatrix mat) = ColMatrix $ Vec.map (\(col, row, val) -> (row, col, val)) mat

shift :: SparseMatrix -> SparseMatrix
shift (RowMatrix mat) = ColMatrix $ Vec.modify (Introsort.sortBy $ comparing extractCol) mat
shift (ColMatrix mat) = RowMatrix $ Vec.modify (Introsort.sortBy $ comparing extractRow) mat

extractRow (row, col, val) = row
extractCol (row, col, val) = col

-- | Multiplies a RowMatrix by a ColMatrix (converting as necessary beforehand)
-- and generates a RowMatrix as a result
mult :: SparseMatrix -> SparseMatrix -> SparseMatrix
mult left@(ColMatrix _) right@(ColMatrix _) = mult (shift left) right
mult left@(ColMatrix _) right@(RowMatrix _) = mult (shift left) (shift right)
mult left@(RowMatrix _) right@(RowMatrix _) = mult left (shift right)
mult (RowMatrix !left) (ColMatrix !right) =
  RowMatrix $ Vec.fromList [ (rowid, colid, multLines 0.0 row col)
    | (rowid, row) <- rows,
      (colid, col) <- cols]
  where
    rows = zip [0..] (getLines extractRow left)
    cols = zip [0..] (getLines extractCol right)

-- | Multiplies a RowMatrix by a ColMatrix (converting as necessary beforehand)
-- and generates a ColMatrix as a result
multCol :: SparseMatrix -> SparseMatrix -> SparseMatrix
multCol left@(ColMatrix _) right@(ColMatrix _) = multCol (shift left) right
multCol left@(ColMatrix _) right@(RowMatrix _) = multCol (shift left) (shift right)
multCol left@(RowMatrix _) right@(RowMatrix _) = multCol left (shift right)
multCol (RowMatrix left) (ColMatrix right) =
  ColMatrix $ Vec.fromList [ (rowid, colid, multLines 0.0 row col)
    | (colid, col) <- zip [0..] (getLines extractCol right),
      (rowid, row) <- zip [0..] (getLines extractRow left)]



--
--
--   INTERNAL METHODS
--
--

multLines :: Double -> [Coord] -> [Coord] -> Double
multLines x [] _ = x
multLines x _ [] = x
multLines !total xs@((_, xcol, xval):xr) ys@((yrow, _, yval):yr) =
  case compare xcol yrow of
    LT -> multLines total xr ys
    GT -> multLines total xs yr
    EQ -> multLines (total + xval * yval) xr yr

getLines :: (Coord -> Int) -> Coords -> [[Coord]]
getLines extract mat
  | Vec.null mat = []
  | otherwise = Vec.toList thisrow : getLines extract rest
  where
    (thisrow, rest) = Vec.span (\c -> extract c == target) mat
    target = extract $ Vec.head mat

-- Gets the index that a row starts at (but it works with cols too)
csrRowBeginsAt :: (Coord -> Int) -> Coords -> Vec.Vector Int
csrRowBeginsAt extract mat =
  Vec.cons 0 $ Vec.accumulate max zeros $ Vec.imap (\idx coord -> (extract coord, idx+1)) mat
  where
    zeros = Vec.replicate (end+1) 0
    end = extract $ Vec.last mat
