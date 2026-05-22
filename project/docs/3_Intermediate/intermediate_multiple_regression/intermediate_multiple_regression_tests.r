library(digest)
library(testthat)
library(tidyverse)



test_1 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_1), 'f7b0db1c9bc01deadcdde41033af1311')
  })
  print("Success!")
}

test_4 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg_LESS), "cf1829ebfea06a34d670471c16367bc7")
  })
  print("Success!")
}


test_5 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg_HS), "ca0ef80393a482fe6a299b69ab9374f8")
  })
  print("Success!")
}

test_6 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg_NU), "dfadec6bab303513637fdf70e319f5a1")
  })
  print("Success!")
}


test_7 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg_U), "3d2023189710af9b7b6679f7ded2e880")
  })
  print("Success!")
}

test_8 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg2), "b9882924c973123dbe67191213a17319")
  })
  print("Success!")
}

test_14 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg3), "b6a9e5ca6bf48bb4f94ac1e2c7d32c07")
  })
  print("Success!")
}

test_17 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg4), "972ba3db121e78411e75e6fee893377d")
  })
  print("Success!")
}
test_20 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg5), "02210f5126c5ea69e4cd850da0ddcad1")
  })
  print("Success!")
}