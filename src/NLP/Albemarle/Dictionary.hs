{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module NLP.Albemarle.Dictionary
    ( apply
    , assignIDs
    , count
    , countAdv
    , discover
    , discoverAdv
    , width
    ) where
import ClassyPrelude
import NLP.Albemarle
import qualified Data.HashSet as HashSet
import qualified Data.HashMap.Strict as HashMap
import qualified System.IO.Streams as Streams
import qualified Data.Vector.Unboxed as Vec
import Debug.Trace
import System.IO.Streams (Generator, InputStream, OutputStream)

type Dictionary = HashMap Text Int
type Histogram = HashMap Text Int

-- | Convert a list of words into a sparse matrix of ID's given a dictionary
apply :: Dictionary -> InputStream [Text] -> IO (InputStream BagOfWords)
apply dict = Streams.map $ sparseMapToVec . HashMap.fromListWith (+) . map (\tok -> (HashMap.lookupDefault 0 tok dict, 1))
  where
    sparseMapToVec = Vec.fromList . sort . HashMap.toList

-- | Count words and then assign them IDs (a convenience method)
discover :: InputStream [Text] -> IO Dictionary
discover docstream = assignIDs <$> count docstream

width :: Dictionary -> Int
width = (1+) . foldl' max 0 . HashMap.elems

-- | Count words and then assign them IDs (a convenience method)
discoverAdv :: Int -> Double -> Int -> Int -> InputStream [Text] -> IO Dictionary
discoverAdv hepax stopword blocksize upper_limit documents =
  assignIDs <$> countAdv hepax stopword blocksize upper_limit documents

-- | Assign IDs to all the words in a dictionary
assignIDs :: Histogram -> Dictionary
assignIDs counts = HashMap.fromList $ zip (sort $ HashMap.keys counts) [1..]

-- | Convenience method: Create a dictionary using the default settings:
--
--   Tokens must be found at least 5 times but not more than 50% of documents
--
--   Documents are handled in blocks of 100 thousand.
--
--   The minimum number of times a token must be found is incrmented when the
--   number of unique filtered tokens reaches over 40 million.
count :: InputStream [Text] -> IO Histogram
count = discoverAdv 5 0.5 100000 40000000

-- | Create a simple dictionary as a HashMap, counting in how many documents
--   each term can be found.
--   Terms that are too rare or too common are removed.
--
--   > count 5 0.5 100000 40000000 $ getTheTextsSomehow
--
--   Means that:
--
--     * terms are in >= 5 instances (hepax threshold)
--     * in <= 50% of documents     (stopword threshold)
--     * in blocks of 100000         (thresholds apply to blocks individually)
--     * incrementing the hepax threshold when the dictionary gets longer than 40M
countAdv :: Int -> Double -> Int -> Int -> InputStream [Text] -> IO Histogram
countAdv hepax stopword blocksize upper_limit documents =
  Streams.map uniqueWords documents -- documents go in here, come out as word counts
    >>= Streams.chunkList blocksize -- Work with 100k documents at a time for memory's sake
    >>= Streams.map (foldl' sumDict mempty) -- For each block: merge the maps, which have (word -> doc count) relations
    >>= Streams.map goldilocks -- For each block: remove too rare and too common words
    >>= Streams.fold sumDict mempty -- Add together the dictionaries from all the blocks

  where
    goldilocks :: Histogram -> Histogram
    goldilocks block = HashMap.filter (\x -> x >= hepax && x <= limit) block
      where limit = floor ( stopword * fromIntegral (HashMap.lookupDefault 1 "" block) )

    -- This trick helps to count the documents in a block without extra
    -- indirections
    uniqueWords :: [Text] -> Histogram
    uniqueWords = HashMap.insert "" 1 .
      HashMap.map (const 1) .
      HashSet.toMap .
      HashSet.fromList

    sumDict :: Histogram -> Histogram -> Histogram
    sumDict = HashMap.unionWith (+)
