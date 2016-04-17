{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
module NLP.Albemarle.Tokens
    ( wordTokenize
    ) where
import Data.Text (Text)
import qualified Data.Text as Text

-- | Tokenize text according to the best method available
wordTokenize :: Text -> [Text]
wordTokenize = whitespaceTokenize

-- | Tokenize strings idiotically, according to the whitespace
whitespaceTokenize :: Text -> [Text]
whitespaceTokenize = Text.words
