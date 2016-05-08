---
layout: post
title:  "Welcome to Jekyll"
date:   2016-05-05 06:28:35 -0400
categories: jekyll update
---
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

```haskell
{-# LANGUAGE OverloadedStrings, NoImplicitPrelude #-}
import ClassyPrelude
import qualified Data.Text as Text
import qualified Data.ByteString as Bytes
import qualified Network.Download as Download
import NLP.Albemarle (Examples, Scrape, Tokens, Phrases, LSI)

main = do
  -- Lines in a (many GB) text file, autodetecting encoding
  let many_docs = sourceFile "mystuff.txt" $= Scrape.decode =$= Scrape.safeLines
  -- Whole files in a directory (False = don't follow symlinks)
  let little_docs = sourceDirectoryDeep False "my_directory" =$= mapC Scrape.decode
  -- Remote documents as well
  let one_more_doc = Scrape.quickURL "http://google.com"
  -- And a PDF for fun
  let a_pdf = sourceFile "my.pdf" $= Scrape.quickPDF

  let docs_as_text = (Scrape.fromHTML remote_stuff) : many_docs
  let phrase_model = Phrases.find $$ mapC Tokens.toWords =$ docs_as_text
  let topic_model = LSI.generate $$ Phrase.use phrase_model <$> docs_as_text
  print topic_model   -- Prints a human-readable summary of the topics

-- TODO
-- Chunking / Shallow Parsing
-- Dependency parsing
```

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
