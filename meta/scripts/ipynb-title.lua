-- Clean up raw YAML cell in converted .ipynb files
-- Extracts only title, author, and date - removes all other metadata

function RawBlock(elem)
  -- Only process ipynb format
  if FORMAT ~= "ipynb" then
    return elem
  end
  
  local text = elem.text
  
  -- Check if this is YAML frontmatter (starts with ---, has title:)
  if not (text:match("^%s*%-%-%-") and text:match("title:")) then
    return elem
  end
  
  -- Extract title
  local title = text:match("title:%s*[\"']([^\"']+)[\"']")
  if not title then
    title = text:match("title:%s*([^\n]+)")
  end
  
  -- Extract author (may span multiple lines or have HTML)
  local author = text:match("author:%s*([^\n]+)")
  
  -- Extract date
  local date = text:match("date:%s*[\"']([^\"']+)[\"']")
  if not date then
    date = text:match("date:%s*([^\n]+)")
  end
  
  -- Clean up extracted values
  if title then
    title = title:gsub("^%s+", ""):gsub("%s+$", "")
  end
  
  if author then
    -- Remove HTML tags like <br>, <br/>, etc.
    author = author:gsub("<br[^>]*>", " ")
    author = author:gsub("<BR[^>]*>", " ")
    -- Remove underscores (markdown italic markers)
    author = author:gsub("_", "")
    -- Remove asterisks
    author = author:gsub("%*", "")
    -- Clean up spaces
    author = author:gsub("%s+", " ")
    author = author:gsub("^%s+", ""):gsub("%s+$", "")
  end
  
  if date then
    date = date:gsub("^%s+", ""):gsub("%s+$", "")
    date = date:gsub("'", "")
  end
  
  -- Build the new YAML content with ONLY title, author, date
  local lines = {"---"}
  
  if title then
    table.insert(lines, 'title: "' .. title .. '"')
  end
  
  if author then
    table.insert(lines, 'author: "' .. author .. '"')
  end
  
  if date then
    table.insert(lines, 'date: "' .. date .. '"')
  end
  
  table.insert(lines, "---")
  
  -- Join lines and return as new RawBlock
  local new_text = table.concat(lines, "\n")
  return pandoc.RawBlock(elem.format, new_text)
end
