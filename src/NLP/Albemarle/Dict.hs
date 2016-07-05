{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE BangPatterns      #-}
{-| A simple, full dictionary, which maintains:

(1) A bijection of words and IDs (one to one);
(2) A count of every word
(3) Methods for keeping memory use under control using filtering

ID 0 is special in this implementation and represents the count and response of
the unknown word. This has some advantages:

(1) Remapping words after they are filtered is easy: map them to unknown, aka 0
(2) The lookup function is total (no undefined or Nothing to handle)
(3) It serves as a nice default (it looks like background noise)
(4) If you don't want that information, use (Vector.drop 1 vec) to eliminate it,
    or use ((Vector.//) vec [(0,0)]) to 0 it out.
|-}
module NLP.Albemarle.Dict
    ( -- * Using Dictionaries
    Dict(..)
    , counts
    , ids
    , hist
    , Histogram(..)
    , countOf
    , idOf
    , select
    , selectMatrix
    , shift
    , size
    , asSparseMatrix
    , asSparseVector
    , asDenseVector

    -- * Creating Dictionaries
    , dictify
    , dictifyAllWords
    , dictifyFirstWords

    -- * Modifying Dictionaries
    , filterDict
    , union
    ) where
import ClassyPrelude hiding (union)
import Control.Monad.ST
import NLP.Albemarle
import qualified Data.Vector.Instances
import qualified Data.Vector.Generic as Vec
import Data.Vector.Generic ((//))
import qualified Data.Vector as BVec
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Algorithms.Search as VSearch
import qualified Data.Vector.Algorithms.Intro as VSort
import qualified Data.List.Ordered as OList


import Data.Text (Text)
import qualified Data.Text as Text
import Data.HashSet (HashSet)
import qualified Data.HashSet as HashSet
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HashMap
import System.IO.Streams (InputStream, OutputStream)
import qualified System.IO.Streams as Streams
import Numeric.LinearAlgebra (Matrix)
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Numeric.LinearAlgebra.Data as HMatrix
import Lens.Micro.TH
import Lens.Micro
import Data.Tuple (swap)

newtype Histogram = Histogram {
  _hist :: HashMap Text Int
} deriving (Show, Eq)
type Ids = BVec.Vector Text
data Dict = Dict {
  _counts :: !Histogram,
  _ids :: !Ids
} deriving (Show, Eq)

makeLenses ''Dict
makeLenses ''Histogram

instance Monoid Histogram where
  mempty = Histogram $ HashMap.singleton "" 0
  mappend (Histogram l) (Histogram r) = Histogram $ HashMap.unionWith (+) l r
instance Semigroup Histogram where
  (<>) = mappend

instance Monoid Dict where
  mempty = Dict {_counts=mempty, _ids=Vec.singleton ""}
  mappend = union
instance Semigroup Dict where
  (<>) = mappend

sortNewV v = Vec.modify VSort.sort $ Vec.fromList v

-- | Filter the dictionary: remove stopwords and hepaxes.
filterDict :: Int   -- ^ the minimum word occurences (to remove rare words)
  -> Float  -- ^ the maximum proportion (0,1] of a word (to remove stopwords)
  -> Int    -- ^ the maximum number of unique words total (to cap memory)
  -> Dict   -- ^ the original dictionary
  -> Dict   -- ^ new dictionary
filterDict min_thresh max_fraction max_words dict = let
  max_thresh = floor $ max_fraction * fromIntegral
    ( sum
    $ HashMap.elems
    $ dict^.counts.hist )
  new_counts =
    HashMap.fromList
    -- This makes sure we preserve the unknown if we can, otherwise 0.
    $ cons ("", HashMap.lookupDefault 0 "" $ dict^.counts.hist)
    $ take max_words
    $ sortOn (negate.snd)
    $ HashMap.toList
    $ HashMap.filter (\v -> min_thresh<=v && v<=max_thresh)
    $ dict^.counts.hist
  in Dict {
    _counts = Histogram new_counts,
    _ids = sortNewV $ HashMap.keys new_counts
  }

-- | Get from a dictionary how many times a word was encountered.
--   What exactly this means depends on if you train the dictionary with
--   dictifyAllWords or dictifyFirstWords.
--   Unknown words return a 0. Keep in mind that on account of filterDict this
--   may not mean it doesn't exist in the corpus; it might have been trimmed
--   because it was too common, too rare, or there wasn't enough memory
countOf :: Dict -> Text -> Int
countOf dict word = fromMaybe 0 $ HashMap.lookup word $ dict^.counts.hist

-- | Get the ID of a word from a Dictionary. Unknown words get a 0 (by design).
idOf :: Dict -> Text -> Int
idOf Dict{_ids=vec} word = runST $ do
  mvec <- Vec.thaw vec
  idx <- VSearch.binarySearch mvec word -- Why is this not pure?!
  return $! maybe 0 (\it-> if it == word then idx else 0) $ (Vec.!?) vec idx

-- | Make a sparse vector representing one document using this dict.
--
--   The length of the vector is the size of the vocabulary, plus one. Keep in
--   mind that the length is determined at runtime so pay attention that you do
--   not use vectors from different dictionaries together lest you cause
--   runtime errors.
asSparseVector :: Dict -> [Text] -> SparseVector
asSparseVector dict = SparseVector (Vec.length $ dict^.ids)
  . sort -- Strictly speaking this isn't always necessary but it is for LSA
  -- . cons (0,0.0) -- This shows we count from 0 (it just happens to be 0.0 there)
  . HashMap.toList
  . HashMap.fromListWith (+)
  . fmap (\w-> (idOf dict w, 1.0))

-- | Make a sparse matrix representing one document using this dict.
--
--   The width of the matrix is the size of the vocabulary, plus one. Keep in
--   mind that the length is determined at runtime so pay attention that you do
--   not use matrices from different dictionaries together lest you cause
--   runtime errors.
--
--   This is equavalent to but faster than repeated SparseVectors because it
--   builds a Hashmap first [O(v)] and then uses that [O(1)/lookup] rather than
--   doing no preparation [O(0)] and repeatedly using binary search [O(log v)]
--   Usually this is a 10-fold or better speedup.
asSparseMatrix :: Dict -> [[Text]] -> SparseMatrix
asSparseMatrix dict = let
  toid word = HashMap.lookupDefault 0 word           -- this evalues many times
    $ HashMap.fromList $ zip (v2l $ dict^.ids) [0..] -- this evaluates once
  tovec = SparseVector (Vec.length $ dict^.ids)
    . sort -- Strictly speaking this isn't always necessary but it is for LSA
    -- . cons (0,0.0) -- This shows we count from 0 (it just happens to be 0.0 there)
    . HashMap.toList
    . HashMap.fromListWith (+)
    . fmap (\w-> (toid w, 1.0))
  in SparseMatrix (Vec.length $ dict^.ids)
    . fmap tovec

-- | Make a dense vector representing one document using this dict.
--
--   The length of the vector is the size of the vocabulary, plus one. Keep in
--   mind that the length is determined at runtime so pay attention that you do
--   not use vectors from different dictionaries together lest you cause
--   runtime errors.
asDenseVector :: Dict -> [Text] -> DenseVector
asDenseVector dict =
  Vec.accum (+) (Vec.replicate (Vec.length $ dict^.ids) 0)
  . fmap (\w-> (idOf dict w, 1.0))

-- | O(n + v log v) Create a dictionary from a single document
--   (also consider the [[Text]] -> Dictionary alternatives, they may be faster)
dictify :: [Text] -> Dict
dictify wds = Dict {
    _counts = Histogram cts,
    _ids = sortNewV $ HashMap.keys cts
  } where
    cts = HashMap.fromListWith (+)
      $ cons ("", 0) -- This is to preserve the 0-is-nothing property
      $ (\x -> (x, 1)) <$> wds

-- | Create a dictionary from many documents, where every instance of every word
--   counts. (As opposed to dictifyFirstWords.) Might be faster than FirstWords.
dictifyAllWords :: [[Text]] -> Dict
dictifyAllWords = dictify . mconcat

-- | Create a dictionary from many documents, counting only the first instance
--   of each word in a given document. Might be slower than AllWords.
dictifyFirstWords :: [[Text]] -> Dict
dictifyFirstWords = dictifyAllWords . fmap (HashSet.toList . HashSet.fromList)

-- | O(1) Retrieve the number of unique words in the vocabulary
size :: Dict -> Int
size dict = HashMap.size $ dict^.counts.hist

-- | O(v) Retrieve the total number of (non-unique) words processed
support :: Dict -> Int
support dict = HashMap.foldl' (+) 0 $ dict^.counts.hist

-- Just for internal use. It's polymorphic, so don't use the eta reduction
v2l a = Vec.toList a
l2v a = Vec.fromList a

-- | What indices into the old matrix will generate an approximation of the new
--   matrix? This is intended for something like
-- > ((HMatrix.?) old_matrix $ select remap) == something like new_matrix
--
--   It's useful for merging termdoc-matrices, SVD's, or the like.
select :: Dict -> Dict -> [Int]
select left right = let
  l = zip [0..] $ v2l $ left^.ids
  r = v2l $ right^.ids
  in select' r l -- backward!

-- | Compute a variant on indirect sorted set intersection. Only used for `select`
select' :: Ord a
  => [a] -- ^ Sorted list X
  -> [(Int, a)] -- ^ Sorted list Y
  -> [Int] -- ^ if X[i] in Y: then Y[Z[i]] = X[i]; else Z[i] = 0
select' [] _  = []
select' ox [] = map (const 0) ox -- Must stay in lock step with X
select' ox@(x:xs) oy@((iy,y):ys) = case compare x y of
  LT -> 0  : select' xs oy
  EQ -> iy : select' xs ys
  GT ->      select' ox ys

-- | Rearrange rows in a matrix to match the order in a new matrix, after a
--   dictionary merge or filter event.
selectMatrix :: Dict -> Dict -> HMatrix.Matrix Double -> HMatrix.Matrix Double
selectMatrix left right old_matrix = (HMatrix.?) old_matrix $ select left right

-- | Given a Remap, convert the original word ID into the new ID
--   (might be more useful curried)
shift :: Dict -> Dict -> Int -> Int
shift left right i = HashMap.lookupDefault 0 i
  $ HashMap.fromList
  $ find'
  (zip [0..] (v2l $ left^.ids))
  (zip [0..] (v2l $ right^.ids))


-- | Compute an indirect sorted set intersection. Only used for `shift`
find' :: Ord a
  => [(Int, a)] -- ^ Left sorted list X
  -> [(Int, a)] -- ^ Right sorted list Y
  -> [(Int, Int)] -- ^ Indices into X, Y representing the sorted intersection
find' [] new  = []
find' orig [] = []
find' ox@((ix,x):xs) oy@((iy,y):ys) = case compare x y of
  LT -> find' xs oy
  EQ -> (ix, iy) : find' xs ys
  GT -> find' ox ys

-- | Union two Dicts (a synonym for mappend). Dict's of huge corpora can be a
--   memory hog, so consider using filterDict occasionally to manage that.
union :: Dict -> Dict -> Dict
union left right = Dict {
    _counts = left^.counts <> right^.counts,
    _ids = Vec.fromList
      $ OList.nub
      $ Vec.toList
      $ Vec.modify VSort.sort (left^.ids <> right^.ids)
  }
