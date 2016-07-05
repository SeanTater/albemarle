{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module NLP.Albemarle.Tokens
    ( icuTokenize
    , whitespaceTokenize
    , wordTokenize
    ) where
import ClassyPrelude
import NLP.Albemarle
import Data.Text (Text)
import Control.Applicative ((<$>))
import qualified Data.Text.ICU as ICU
import qualified Data.Text as Text

-- | Tokenize text according to the best method available
wordTokenize :: Text -> [Text]
wordTokenize = icuTokenize

-- | Tokenize strings idiotically, according to the whitespace
whitespaceTokenize :: Text -> [Text]
whitespaceTokenize = Text.words

-- | Tokenize text according to the Unicode standard, nut filter out spaces.
icuTokenize :: Text -> [Text]
icuTokenize line =
  filter (\x -> not $ x==" ")
  $ ICU.brkBreak
  <$> ICU.breaks (ICU.breakWord ICU.Current) line
