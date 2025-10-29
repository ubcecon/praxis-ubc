-- Quarto QMD to IPYNB Title Formatter
-- 
-- Purpose: Fix broken .ipynb output in the title cell
--
-- Usage: Add to all .qmd headers
--   format:
--     ipynb:
--       filters:
--        - ipynb-title.lua

-- Global storage for metadata
local metadata = {}

-- Meta function: Extracts title, author, and date from YAML
function Meta(meta)
  -- Extract title
  if meta.title then
    metadata.title = pandoc.utils.stringify(meta.title)
  end
  
  -- Extract and process author(s)
  if meta.author then
    if type(meta.author) == "table" then
      -- Handle multiple authors or structured format
      local authors = {}
      for _, author in ipairs(meta.author) do
        if type(author) == "table" and author.name then
          table.insert(authors, pandoc.utils.stringify(author.name))
        else
          table.insert(authors, pandoc.utils.stringify(author))
        end
      end
      if #authors > 0 then
        metadata.author = table.concat(authors, ", ")
      end
    else
      -- Single author
      metadata.author = pandoc.utils.stringify(meta.author)
    end
    
    -- Clean up author string (remove HTML tags, markdown formatting, extra spaces)
    if metadata.author then
      metadata.author = metadata.author:gsub("<[Bb][Rr]%s*/?>", " ")  -- Remove <br> tags
      metadata.author = metadata.author:gsub("_", "")                  -- Remove underscores
      metadata.author = metadata.author:gsub("%*", "")                 -- Remove asterisks
      metadata.author = metadata.author:gsub("%s+", " ")               -- Normalize spaces
      metadata.author = metadata.author:gsub("^%s+", "")               -- Trim leading
      metadata.author = metadata.author:gsub("%s+$", "")               -- Trim trailing
    end
  end
  
  -- Extract date
  if meta.date then
    metadata.date = pandoc.utils.stringify(meta.date)
  end
  
  return meta
end

-- Helper: Create formatted header blocks
local function create_formatted_header()
  local blocks = {}
  
  -- Add title as level 1 heading
  if metadata.title then
    table.insert(blocks, pandoc.Header(1, pandoc.Inlines(metadata.title)))
  end
  
  -- Create author and date line
