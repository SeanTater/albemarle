{-# LANGUAGE BangPatterns #-}
module NLP.Albemarle.GloVe (
  ContextVector,
  TrainingApproach,
  adagrad,
  deemph,
  gradient,
  randomModel,
  train
) where
import Numeric.LinearAlgebra (Matrix, Vector, dot)
import qualified Data.Vector.Storable as SVec
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Numeric.LinearAlgebra.Data as HMatrix
import qualified Data.IntMap as IntMap
import Lens.Micro
import Lens.Micro.TH

-- Constants: these are known probably not worth parameterizing because their
-- current settings seem to work the best in all situations.
deemph_alpha = 0.75 -- Cooccurance frequencies are raised to this power
deemph_cap = 100 -- Cooccurance frequencies are clamped to [1..deemph_cap]
learning_rate = 0.05 -- Adagrad learning rate: it's not super sensitive

-- | One input or output context vector, as well as the sum of squared gradients
--   for each dimension, as needed for adagrad
data ContextVector = ContextVector {
  _bias :: !Double, -- ^ Bias associated with the vector
  _embedding :: !Vector Double, -- ^ The input or output vector
  _biasHist :: !Double, -- ^ The bias's SSE (calling it history) - for adagrad
  _embeddingHist :: !Vector Double -- ^ All other SSE's - for adagrad
}
-- | A function computing a training step. This is merely a synonym for clarity.
type TrainingApproach = ContextVector -> ContextVector -> Double
  -> (ContextVector, ContextVector)
makeLenses ''ContextVector

-- | Measure the gradient of error between two vectors. It's used as part of a
--   numerical optimization. (e.g. Adagrad)
gradient :: ContextVector -- ^ The source context vector
  -> ContextVector -- ^ The target context vector
  -> Double -- ^ Cooccurance count of the two vectors
  -> Double -- ^ The gradient of error between vectors
gradient source target edgefrequency = let
  commonness = dot (source^.embedding) (target^.embedding) -- context vector
  logfreq_error = (source^.bias) * (target^.bias) - log edgefrequency -- bias
  in commonness + logfreq_error -- both sources

-- | Deemphasize high frequencies, to avoid overweighting stopwords
deemph :: Double -> Double
deemph f = min 1 ((f/deemph_cap) ** deemph_alpha)

-- | An adaptation of Adagrad to GloVe. The difference from a typical optimizer
--   is that it is intended to operate on large streaming sources of vector
--   pairs.
adagrad :: ContextVector -- ^ Source vector
  -> ContextVector -- ^ Target vector
  -> Double -- ^ Cooccurance frequency
  -> (ContextVector, ContextVector) -- ^ Modified source and target vectors
adagrad source target edgefrequency history = let
  grad = learning_rate * gradient source target edgefrequency
  source_grad = grad * (source^.embedding)
  target_grad = grad * (target^.embedding)
  repair s t sgrad tgrad = ContextVector {
    _embedding = (s^.embedding) - (tgrad / sqrt (s^.embeddingHist))
    _bias = (s^.bias) - (grad / sqrt (s^.biasHist))
    _embeddingHist = (s^.embeddingHist) + (sgrad * sgrad)
    _biasHist = (s^.biasHist) + (grad * grad)
  }
  in (repair source target source_grad target_grad,
      repair target source target_grad source_grad)

-- | Train a model using a specific training approach, an existing model, and
--   some training cooccurances.
train :: IntMap ContextVector -- ^ The initial parameters
  -> TrainingApproach -- ^ The approach to training (like adagrad)
  -> [(Int, Int, Double)] -- ^ (Probably lazy) list of cooccurances
  -> IntMap ContextVector -- ^ Trained model
train trainer params [] = params
train trainer !params (s,t,v):insts = let
  get = (IntMap.!)
  (reps, rept) = trainer (get s) (get t) v
  in train trainer (IntMap.insert s reps $ IntMap.insert t rept params) insts

-- | Make a model out of random vectors (for bootstrapping)
randomModel :: Int -- | Number of words
  -> Int -- | Vector length
  -> IO (IntMap ContextVector) -- | Resulting map
randomModel height width = do
  mat <- HMatrix.rand height width
  return $! IntMap.fromList $ zip [0..] $ HMatrix.toRows mat
