-- Fix quarto convert titles in .ipynb for website launch menu
-- Works by removing the raw YAML frontmatter cells that appear when converting to Jupyter notebooks

function RawBlock(elem)
  if FORMAT == "ipynb" then -- only on .ipynb
    local text = elem.text:gsub("^%s+", ""):gsub("%s+$", "") -- trim whitespace on raw cells
    
    -- Pattern: starts with ---, contains key: value pairs, ends with ---
    if text:match("^%-%-%-") and text:match("%-%-%-[%s\n]*$") and text:match("%w+:%s*[^\n\r]+") then
      return {}
    end
  end
  
  return elem
end
