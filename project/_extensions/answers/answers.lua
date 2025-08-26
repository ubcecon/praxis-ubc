local function has_value (tab, val)
  for index, value in ipairs(tab) do
      if value == val then
          return true
      end
  end

  return false
end

function ModifyDivs(el)
  -- Specify the class to silence
  local silencedClass = "answer"

  -- quarto.log.output(el)
  -- quarto.log.output(has_value(el.classes, silencedClass))
  -- Check if the <div> tag has the specified class
  if  el.attributes['tags'] and el.attributes['tags'] == silencedClass then

    quarto.log.output("Purging answers")
    -- Set the content of the <div> to an empty table
    el = {}
    return el
  end
  if has_value(el.classes, silencedClass) then

      quarto.log.output("Purging answers")
      -- Set the content of the <div> to an empty table
      el = {}
      return el
  end
  
end

return {
  {
    Div = ModifyDivs
  }

}