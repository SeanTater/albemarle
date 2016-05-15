{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
-- | This is an example.
--   Yes, it is.
module NLP.Albemarle.Dictionary
    ( discover
    , discoverAdv
    ) where
import ClassyPrelude
import qualified Data.HashSet as HashSet
import qualified Data.HashMap.Strict as HashMap
import qualified System.IO.Streams as Streams
import System.IO.Streams (Generator, InputStream, OutputStream)

-- | Convenience method: Create a dictionary using the default settings:
--
--   Tokens must be found at least 5 times but not more than 50% of documents
--
--   Documents are handled in blocks of 100 thousand.
--
--   The minimum number of times a token must be found is incrmented when the
--   number of unique filtered tokens reaches over 40 million.
discover :: InputStream [Text] -> IO (HashMap Text Int)
discover = discoverAdv 5 0.5 100000 40000000

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
discoverAdv :: Int -> Double -> Int -> Int -> InputStream [Text] -> IO (HashMap Text Int)
discoverAdv hepax stopword blocksize upper_limit documents =
  Streams.map uniqueWords documents >>= -- documents go in here, come out as word counts
  Streams.chunkList blocksize >>= -- Work with 100k documents at a time for memory's sake
  Streams.map (foldl' sumDict mempty) >>= -- For each block: merge the maps, which have (word -> doc count) relations
  Streams.map goldilocks >>= -- For each block: remove too rare and too common words
  Streams.fold sumDict mempty


  --sumDict $ -- Sum together all the blocks' useful words
  --goldilocks <$> -- For each block: remove too rare and too common words
  --sumDict <$>  -- For each block: merge the maps, which have (word -> doc count) relations
  --blockify  -- Work with 100k documents at a time for memory's sake
  --(uniqueWords <$> documents)

  where
    goldilocks :: HashMap Text Int -> HashMap Text Int
    goldilocks block = HashMap.filter (\x -> x >= hepax && x <= limit) block
      where limit = floor ( stopword * fromIntegral (HashMap.lookupDefault 1 "" block) )

    -- This trick helps to count the documents in a block without extra
    -- indirections
    uniqueWords :: [Text] -> HashMap Text Int
    uniqueWords = HashMap.insert "" 1 .
      HashMap.map (const 1) .
      HashSet.toMap .
      HashSet.fromList

    sumDict :: HashMap Text Int -> HashMap Text Int -> HashMap Text Int
    sumDict = HashMap.unionWith (+)
