library(testthat)
library(digest)

test_1 <- function() {
  test_that("Solution is incorrect. Hint: remember the interaction variables?", {
    expect_equal(digest(reg0$coefficients["(Intercept)"]), "2757bf99da617aba72b05eb12b01ddee")
  })
  print("Success!")
}

test_2 <- function() {
  test_that("Solution is incorrect. Hint: remember the interaction variables?", {
    expect_equal(digest(reg1$coefficients["(Intercept)"]), "96d309e48f9af503eee74def18c9faef")
  })
  print("Success!")
}

test_3 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg2$coefficients["(Intercept)"]), "374f126161a74b2f63b13bcc1fc46f24")
  })
  print("Success!")
}
test_4 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg3A$coefficients["(Intercept)"]), '5277518dfdc0f4b416ec7cbf60f20916')
  })
  print("Success!")
}

test_5 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg3M$coefficients["(Intercept)"]), '093aa84e9899b2118c20e1c97ee54f0d')
  })
  print("Success!")
}

test_6 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(reg3F$coefficients["(Intercept)"]), "833c0bd837531e1c7803ebe7942649f9")
  })
  print("Success!")
}

test_7 <- function() {
  test_that("Solution is incorrect. Hint: interactions?", {
    expect_equal(digest(reg4$coefficients["(Intercept)"]), "327f4497a51c0745f77a486a3724289a")
  })
  print("Success!")
}
