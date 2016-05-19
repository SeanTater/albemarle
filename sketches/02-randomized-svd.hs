#!/usr/bin/env runhaskell
{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
import ClassyPrelude hiding ((<>)) -- this <> is for semigroups (usually better)
import Data.Monoid ((<>)) -- ... but HMatrix only has Monoid
import Numeric.LinearAlgebra (tr) -- transpose
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Data.Text.Format as Format

main = do
  -- Load the term-document matrix
  putStrLn "Loading"
  matrix <- HMatrix.loadMatrix "termdoc.txt"
  result <- stochasticTruncatedSVD 50 2 matrix
  print $ HMatrix.size result

-- Stochastic SVD. See Halko, Martinsson, and Tropp, 2010 for an explanation
stochasticTruncatedSVD :: Int -> Int -> HMatrix.Matrix Double -> IO (HMatrix.Matrix Double)
stochasticTruncatedSVD top_vectors num_iterations original = do
  let (m, n) = HMatrix.size original
  let k = top_vectors + 100
  -- Stage A
  putStrLn "Stage A\n\tPower Iteration"
  omega <- HMatrix.randn n k
  let
    y 0 = original <> omega
    y n = seq (y (n-1)) $ original <> tr original <> y (n-1)
  bigY <- return $! y num_iterations -- Force evaluation here
  putStrLn "\tOrthogonalizing"
  q <- return $! HMatrix.orth bigY

  -- Stage B
  putStrLn "Stage B"
  let b = tr q <> original
      (uhat, sigma, vt) = HMatrix.thinSVD b
      u = q <> uhat

  return $! HMatrix.subMatrix (0,0) (n, top_vectors) $ tr vt
  --putStrLn "Checking"
  --let ahat = u <> HMatrix.diag sigma <> tr vt

  --Format.print "MSE of approximation {}\n" [HMatrix.sumElements ((a-ahat)^2) / (fromIntegral $ m*n)]
