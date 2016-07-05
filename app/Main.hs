{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, BangPatterns #-}
module Main where
import ClassyPrelude
import qualified Numeric.LinearAlgebra as HMatrix
import NLP.Albemarle.Dict (counts, ids, hist)
import qualified NLP.Albemarle.Dict as Dict
import NLP.Albemarle.LSA (termvectors, topicweights)
import qualified NLP.Albemarle.LSA as LSA
import qualified NLP.Albemarle.Tokens as Tokens
import qualified System.IO.Streams as Streams
import Data.Text.Encoding.Error (lenientDecode)
import Lens.Micro

main :: IO ()
main = do
  (final_dict, model) <- Streams.withFileAsInput "kjv.verses.txt.gz" $ \file ->
    Streams.gunzip file
    >>= Streams.lines
    >>= Streams.decodeUtf8With lenientDecode
    >>= Streams.map Tokens.wordTokenize
    >>= Streams.chunkList 5000 -- Yep, I'm making this stuff up.
    >>= Streams.map (\chunk -> let
      dict = Dict.dictifyAllWords chunk
      sparsem = Dict.asSparseMatrix dict chunk
      lsa = LSA.lsa 100 sparsem
      in (dict, lsa))
    >>= Streams.fold (\(!d1, !lsa1) (d2, lsa2) -> let
      d3 = Dict.filterDict 2 0.5 1000 $ d1 <> d2
      lsa3 = LSA.rebase d1 d3 lsa1 <> LSA.rebase d2 d3 lsa2
      in (d3, lsa3)) (mempty, mempty)

  -- It should use all 100 words allowed plus the unknown
  print $ length (final_dict^.counts.hist)
  -- It should have a full size LSA as well
  print $ HMatrix.size (model^.termvectors)
  -- Plus topic weights
  print $ HMatrix.size (model^.topicweights)
