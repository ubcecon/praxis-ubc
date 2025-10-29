-- ipynb-title.lua
-- Quarto QMD to IPYNB Title Formatter
-- 
-- Purpose: Removes ugly raw YAML frontmatter from converted Jupyter notebooks
--          and replaces it with a clean, formatted title, author, and date
--
-- Usage: Add to your .qmd YAML header:
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
  local info_elements = {}
  
  if metadata.author then
    table.insert(info_elements, pandoc.Str(metadata.author))
  end
  
  if metadata.date then
    if #info_elements > 0 then
      table.insert(info_elements, pandoc.Space())
      table.insert(info_elements, pandoc.Str("•"))
      table.insert(info_elements, pandoc.Space())
    end
    table.insert(info_elements, pandoc.Str(metadata.date))
  end
  
  if #info_elements > 0 then
    table.insert(blocks, pandoc.Para(info_elements))
  end
  
  -- Add horizontal rule separator
  if #blocks > 0 then
    table.insert(blocks, pandoc.HorizontalRule())
  end
  
  return blocks
end

-- Helper: Check if block is YAML frontmatter
local function is_yaml_frontmatter(block)
  if block.t ~= "RawBlock" then
    return false
  end
  
  local text = block.text:gsub("^%s+", ""):gsub("%s+$", "")
  
  -- Check for YAML pattern: starts with ---, has key:value pairs, ends with ---
  return text:match("^%-%-%-") and 
         text:match("%-%-%-[%s\n]*$") and 
         text:match("%w+:%s*[^%\n%\r]+")
end

-- Pandoc function: Main document processor
function Pandoc(doc)
  -- Only process ipynb format
  if FORMAT ~= "ipynb" then
    return doc
  end
  
  local new_blocks = {}
  local found_yaml = false
  local inserted_header = false
  
  -- Process each block
  for i, block in ipairs(doc.blocks) do
    if is_yaml_frontmatter(block) then
      -- Remove YAML frontmatter block
      found_yaml = true
    else
      -- Insert formatted header before first content block
      if found_yaml and not inserted_header then
        local header_blocks = create_formatted_header()
        for _, hblock in ipairs(header_blocks) do
          table.insert(new_blocks, hblock)
        end
        inserted_header = true
      end
      
      -- Keep the current block
      table.insert(new_blocks, block)
    end
  end
  
  -- Handle case where YAML is the last block
  if found_yaml and not inserted_header then
    local header_blocks = create_formatted_header()
    for _, hblock in ipairs(header_blocks) do
      table.insert(new_blocks, hblock)
    end
  end
  
  return pandoc.Pandoc(new_blocks, doc.meta)
end
