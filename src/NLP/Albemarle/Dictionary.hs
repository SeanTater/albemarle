{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module NLP.Albemarle.Dictionary
    ( count
    ) where
import ClassyPrelude.Conduit
import qualified Data.HashSet as HashSet
import qualified Data.HashMap.Strict as HashMap

-- ! Create a simple dictionary as a HashMap, counting unique tokens
-- ! Remove words that appear too few or too many times per block.
-- ! For example:
-- ! count 5 0.5 100000 $ getTheTextsSomehow
-- ! Means to work in blocks of 100k documents, keeping tokens found at least
-- ! 5 times but found in at most 50% of documents
count :: Int -> Double -> Int -> [[Text]] -> HashMap Text Int
count hepax stopword blocksize documents =
  sumDict $ -- Sum together all the blocks' useful words
  goldilocks <$> -- For each block: remove too rare and too common words
  sumDict <$>  -- For each block: merge the maps, which have (word -> doc count) relations
  blockify  -- Work with 100k documents at a time for memory's sake
  (uniqueWords <$> documents)

  where
    goldilocks :: HashMap Text Int -> HashMap Text Int
    goldilocks block = HashMap.filter (\x -> x >= hepax && x <= limit) block
      where limit = floor ( stopword * fromIntegral (HashMap.lookupDefault 1 "" block) )

    blockify :: [HashMap Text Int] -> [[HashMap Text Int]]
    blockify [] = []
    blockify docs = start : blockify rest
      where (start, rest) = splitAt blocksize docs

    -- This trick helps to count the documents in a block without extra
    -- indirections
    uniqueWords :: [Text] -> HashMap Text Int
    uniqueWords = HashMap.insert "" 1 .
      HashMap.map (const 1) .
      HashSet.toMap .
      HashSet.fromList

    sumDict :: [HashMap Text Int] -> HashMap Text Int
    sumDict = foldl' (HashMap.unionWith (+)) mempty
