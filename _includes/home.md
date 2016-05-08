# What we're shooting for
Albemarle is an open source natural language processing and topic modeling
library intended to strike a balance between ease of use and reliability.
Our goal is for you to run it on tons of crazy and sometimes broken data like
web crawls and product databases, but still get the best response you can.
So in the future, we want you to be able to use an API like so:

```haskell
{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
import           ClassyPrelude
import           Control.Monad ((>=>))
import qualified System.IO.Streams as Streams
import           NLP.Albemarle (Scrape, Tokens, Phrases, LSI)
main = do
  -- Lines in a (many GB) text file, autodetecting encoding
  let many_docs = Streams.withFileAsInput "mystuff.txt.gz"
                  >=> Streams.gunzip >=> Scrape.decode >=> Scrape.safeLines
  -- Whole files in a directory (False = don't follow symlinks)
  let little_docs = Scrape.glob "my_directory/*.txt" >=> Streams.map Scrape.decode
  -- Remote documents as well
  let one_more_doc = Scrape.url "http://google.com" >=> Streams.map Scrape.html
  -- And a PDF for fun
  let a_pdf = Streams.withFileAsInput "my.pdf" >=> Streams.map Scrape.quickPDF

  let all_the_docs = Streams.concurrentMerge [many_docs, little_docs, one_more_doc, a_pdf]
                     >=> Tokens.toWords
  let phrase_model = Phrases.find all_the_docs
  let topic_model = LSI.generate $ all_the_docs >=> Phrases.use phrase_model
  putStrLn topic_model   -- Prints a human-readable summary of the topics
```

# How far we've gone
You can check out how our [continuous integration builds][builds] are going.

[builds]: https://travis-ci.org/SeanTater/albemarle
