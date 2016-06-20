{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, QuasiQuotes, FlexibleContexts, TemplateHaskell #-}
module NLP.Albemarle.Scrape (
  download,
  readAsText
) where
import ClassyPrelude
import Data.ByteString (ByteString)
import Data.Function ((&))
import Control.Lens hiding (re) -- Regex needs this
import qualified Data.Text.ICU.Convert as Convert
import qualified Data.ByteString as Bytes
import qualified Data.Text as Text
import qualified Network.Curl as Curl
import qualified Network.Curl.Info as Curl
import Text.Regex.PCRE.Heavy

-- A strict in-memory file of some content type we don't know
declareLenses [d|
  data Download = Download {
    mime, enc :: Maybe Text,
    payload :: ByteString
  } deriving (Show, Eq)
  |]
emptyDownload = Download Nothing Nothing ""

-- | Download from a URL, and if possible, trying to straighten out the encoding
--   then return the result as a Text (as opposed to a String)
--   It works by using the content-type header in HTTP, then converts with
--   text-icu.
download :: Text -> IO Download
download url = do
  curlresponse <- Curl.curlGetResponse_ (unpack url) [] :: IO (Curl.CurlResponse_ [(String, String)] ByteString)
  -- !!! code <- Curl.respCurlCode
  resp <- mimeAndEncFromHeader curlresponse
  return $! resp & payload .~ Curl.respBody curlresponse
  where
    mimeAndEncFromHeader curlresponse = do
      Curl.IString contenttype <- Curl.respGetInfo curlresponse Curl.ContentType
      let groups = scan
                    [re|([a-z-/]+)(; ?charset=([a-z0-9-_ ]+))?|]
                    (Text.pack contenttype)
      return $ case groups of
        [(_, [mimetype])] -> emptyDownload & mime .~ Just mimetype
        [(_, [mimetype, _, charset])] -> emptyDownload & mime .~ Just mimetype & enc .~ Just charset
        otherwise -> emptyDownload

readAsText :: Download -> IO Text
readAsText download = do
  let encoding = unpack $ fromMaybe "" (download^.enc)
  converter <- Convert.open encoding (Just False) -- making a converter is IO
  return $! Convert.toUnicode converter (download^.payload)

--readAsHTML download = do
--  return expression

--makeLenses ''Download
