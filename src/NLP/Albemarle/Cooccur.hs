{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
module NLP.Albemarle.Cooccur
  ( -- * Sliding Window
    geometricSkips,
    harmonicSkips,
    packIDs,
    unpackIDs,
    cooccurify,
    wordSlice,
    frequency,
    wordFrequency,
    probability,
    predict
  ) where
import Lens.Micro
import Lens.Micro.TH
import qualified Data.IntMap.Strict as IntMap
import Data.Bits
import Data.Maybe
import Debug.Trace

type Cooccur = IntMap.IntMap Double

-- | Windowing function for skip-grams
geometricSkips :: Double  -- ^ Exponential decay with distance
  -> Int     -- ^ Window radius (actual size is 2*radius + 1)
  -> [a]     -- ^ Document
  -> [(a, a, Double)]  -- ^ (Source, target, weight) triple
geometricSkips dropoff radius [] = []
geometricSkips dropoff radius (s:ss) =
  -- Tack on source and get two tuples each, because it is symmetric
  (concatMap (\(t, w) -> [(s, t, w), (t, s, w)])
  $ zip ss -- Pair the weights with the upcoming words
  $ take radius -- Limit to a horizon
  $ iterate (dropoff*) 1) -- The decaying weights
  ++ geometricSkips dropoff radius ss -- Repeat for the next word

-- | Windowing function for skip-grams
harmonicSkips :: Int     -- ^ Window radius (actual size is 2*radius + 1)
  -> [a]     -- ^ Document
  -> [(a, a, Double)]  -- ^ (Source, target, weight) triple
harmonicSkips radius [] = []
harmonicSkips radius (s:ss) =
  (concatMap (\(t, w) -> [(s, t, w), (t, s, w)])
  $ zip ss
  $ fmap (1/) [1..fromIntegral radius])
  ++ harmonicSkips radius ss

-- | This is dangerous but great for Intmaps: convert two 32 bit integers into
--   a single packed 64 bit integer. But Intmaps can't guarantee 64 bits, so
--   this will not work on 32-bit machines. Be warned!
packIDs :: Int -> Int -> Int
packIDs a b = (a `shiftL` 32) .|. b

-- | This is dangerous but great for Intmaps: convert one 64 bit integer into
--   two 32 bit integers. But Intmaps can't guarantee 64 bits, so
--   this will not work on 32-bit machines. Be warned!
unpackIDs :: Int -> (Int, Int)
unpackIDs a = (a `shiftR` 32, a .&. 0x00000000FFFFFFFF)

-- | Make a word-word cooccurance matrix out of a series of weighted
--   cooccurances (like harmonicSkips will make for you)
cooccurify :: [(Int, Int, Double)] -> Cooccur
cooccurify = IntMap.fromListWith (+)
  . fmap (\(s, t, w) -> (packIDs s t, w))

-- | Get all of the frequencies associated with a source word
wordSlice :: Int -> Cooccur -> Cooccur
wordSlice a = let
  start = packIDs a 0 -- Will be inclusive
  end = packIDs (a+1) 0 -- Will be exclusive
  split target c = let
    (left, mayv, right) = IntMap.splitLookup target c
    in (maybe left (\v -> IntMap.insert target v left) mayv, right)
  in snd . split start -- after the beginning
    . fst . split end -- before the end


-- | Get the frequency of a specific word pair.
frequency :: Int -> Int -> Cooccur -> Double
frequency a b = fromMaybe 0 . IntMap.lookup (packIDs a b)

wordFrequency :: Int -> Cooccur -> Double
wordFrequency wd = sum . fmap snd . IntMap.toList . wordSlice wd

probability :: Int -> Int -> Cooccur -> Double
probability a b cooccur = let
  denom = wordFrequency a cooccur
  in if denom == 0 then 0 else frequency a b cooccur / denom

predict :: Int -> Cooccur -> [(Int, Double)]
predict wd cooccur = let
  normalize = wordFrequency wd cooccur
  in fmap (\(p, f) -> (snd $ unpackIDs p, f/normalize))
    $ IntMap.toList $ wordSlice wd cooccur
