library(testthat)
library(digest)

test_1 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(table_1), '992814670932b572a1d7559e69282b86')
  })
  print("Success!")
}

test_2 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(table_2), '6c6051beee039dec369d9e1c9cdcc9c3')
  })
  print("Success!")
}


test_3 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(table_3), '0da3d775365a43d84d47c6d5107dbdb3')
  })
  print("Success!")
}