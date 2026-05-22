library(testthat)
library(digest)

test_1 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_1), "6e74b7e5365030d8ee0d0bdaee70abdc")
  })
  print("Success!")
}


test_2 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_2), "d0782e9eab2ce97d4e82e8878a7b4cf4")
  })
  print("Success!")
}


test_3 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_3), "4679f693970819ff9d9fd35e1ef900f3")
  })
  print("Success!")
}

test_4 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_4), "77f2bb4b0d574be2739fa0e35b9c673a")
  })
  print("Success!")
}

test_5 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_5), "2d0d61812ea3ae636c44b6d7eff1c719")
  })
  print("Success!")
}

test_6 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_6), "3e3207188056fc819ed90a154a26f40e")
  })
  print("Success!")
}

test_7 <- function(){
  ans <- digest(answer_7)
    case_when(ans == "75f1160e72554f4270c809f041c7a776" ~
    "Remember the standard error is the standard deviation of the sampling distribution, not the population!",
    ans == "3a5505c06543876fe45598b5e5e5195d" ~
    "Correct! The standard error shows the average variation in of the observations in the sampling distribution (the various possible sample means for a given sample size).",
    ans =="475bf9280aab63a82af60791302736f6" ~
    "No, our sample estimate cannot be exactly correct. The standard error gives a hint as to how likely it is that our sample estimate is close to our population parameter.",
    ans =="c1f86f7430df7ddb256980ea6a3b57a4" ~
    "Incorrect, try again!",
    TRUE ~ "Review formatting of your answer")
  }

test_8 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_8), "630f53fab2fb1c26b9feff126ea69459")
  })
  print("Success!")
}

test_9 <- function() {
  test_that("Solution is incorrect", {
    expect_equal(digest(answer_9), "ebe557aaf54dcd977eb624e5fad200f2")
  })
  print("Success!")
}
