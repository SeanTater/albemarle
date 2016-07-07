{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
module NLP.Albemarle.Cooccur
  ( -- * Sliding Window
    decayingSkips,
    packIDs,
    unpackIDs,
    cooccurify
  ) where
import Lens.Micro
import Lens.Micro.TH
import qualified Data.IntMap.Strict as IntMap
import Data.Bits
import Data.Maybe
import Debug.Trace

type Cooccur = IntMap.IntMap Double

-- | Windowing function for skip-grams
decayingSkips :: Double  -- ^ Exponential decay with distance
  -> Int     -- ^ Window radius (actual size is 2*radius + 1)
  -> [a]     -- ^ Words to use a recent history (or use [])
  -> [a]     -- ^ Document
  -> [(a, a, Double)]  -- ^ (Source, target, weight) triple
decayingSkips dropoff radius window [] = []
decayingSkips dropoff radius window (s:ss) =
     (zip3 (repeat s) window $ take radius $ iterate (dropoff*) 1)
  ++ (zip3 (repeat s) ss     $ take radius $ iterate (dropoff*) 1)
  ++ decayingSkips dropoff radius (s : fst (splitAt radius window)) ss

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
--   cooccurances
cooccurify :: [(Int, Int, Double)] -> Cooccur
cooccurify = IntMap.fromListWith (+) . map (\(s, t, w) -> (packIDs s t, w))

-- | Get all of the frequencies associated with a source word
wordSlice :: Int -> Cooccur -> Cooccur
wordSlice a = let
  start = packIDs a 0 -- One before the beginning: split is exclusive
  end = packIDs (a+1) 0 - 1
  split a c = let
    (x, y, z) = IntMap.splitLookup a c
    reconstruct m = maybe m (\v -> IntMap.insert a v m) y
    in (reconstruct x, reconstruct z)
  in snd . split start -- after the beginning
    . fst . split end -- before the end

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
