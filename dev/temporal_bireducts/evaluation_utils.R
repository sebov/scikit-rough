get.stats.1 <- function(result, number.of.cols=NA, params=NA) {
  stopifnot(is.numeric(number.of.cols) || is.list(params))
  if (is.list(params)) {
    number.of.cols <- params$data.pack.cols
  }
  stats <- rep(0, number.of.cols)
  for (r in result$add.series) {
    stats[r$real.cols] <- stats[r$real.cols] + 1
  }
  return(stats)
}



get.stats.2 <- function(result, number.of.cols=NA, params=NA) {
  stopifnot(is.numeric(number.of.cols) || is.list(params))
  if (is.list(params)) {
    number.of.cols <- params$data.pack.cols
  }
  stats <- rep(0, number.of.cols)

  for (r in result$add.series) {
    computed.cols <- r$computed.cols
    real.cols <- r$real.cols
    for (i in 1:length(computed.cols)) {
      if (computed.cols[[i]] > i) {
        stats[[real.cols[[i]]]] <- stats[[real.cols[[i]]]] + computed.cols[[i]] - i
      }
    }
  }
  return(stats)
}



get.stats.3 <- function(result, number.of.cols=NA, params=NA) {
  stopifnot(is.numeric(number.of.cols) || is.list(params))
  if (is.list(params)) {
    number.of.cols <- params$data.pack.cols
  }
  stats <- rep(0, number.of.cols)

  for (r in result$add.series) {
    real.cols <- r$real.cols
    for (i in 1:length(real.cols)) {
      stats[[real.cols[[i]]]] <- stats[[real.cols[[i]]]] + i
    }
  }
  return(stats)
}



get.stats.4 <- function(result, number.of.cols=NA, params=NA) {
  stopifnot(is.numeric(number.of.cols) || is.list(params))
  if (is.list(params)) {
    number.of.cols <- params$data.pack.cols
  }
  stats <- rep(0, number.of.cols)

  for (r in result$add.series) {
    computed.cols <- r$computed.cols
    real.cols <- r$real.cols
    for (i in 1:length(computed.cols)) {
      if (computed.cols[[i]] > i) {
        stats[[real.cols[[i]]]] <- stats[[real.cols[[i]]]] + computed.cols[[i]] - i
      }
    }
    for (i in 1:computed.cols[[length(computed.cols)]]) {
      if (! i %in% computed.cols) {
        stats[[r$current.table.cols.permutation[[i]]]] <- stats[[r$current.table.cols.permutation[[i]]]] - 1
      }
    }
  }
  return(stats)
}


library(sets)

get.stats.5 <- function(result, number.of.cols=NA, params=NA) {
  stopifnot(is.numeric(number.of.cols) || is.list(params))
  if (is.list(params)) {
    number.of.cols <- params$data.pack.cols
  }
  stats <- rep(0, number.of.cols)

  for (r in result$add.series) {
    computed.cols <- r$computed.cols
    real.cols <- r$real.cols
    middle <- ceiling(length(computed.cols) / 2)
    for (i in 1:computed.cols[[length(computed.cols)]]) {
      if (i <= computed.cols[[middle]]) {
        if (! i %in% computed.cols[1:middle]) {
          stats[[r$current.table.cols.permutation[[i]]]] <- stats[[r$current.table.cols.permutation[[i]]]] - 1
      	}
      } else {
        if (i %in% computed.cols[(middle + 1):length(computed.cols)]) {
          stats[[r$current.table.cols.permutation[[i]]]] <- stats[[r$current.table.cols.permutation[[i]]]] + 1
      	}
      }
    }
  }
  return(stats)
}


x1 = get.stats.1(result, params=params)
x2 = get.stats.2(result, params=params)
x3 = get.stats.3(result, params=params)
x4 = get.stats.4(result, params=params)
x5 = get.stats.5(result, params=params)

wilcox.test(x1[1:params$data.pack.important.cols], x1[(params$data.pack.important.cols + 1):params$data.pack.cols])
wilcox.test(x2[1:params$data.pack.important.cols], x2[(params$data.pack.important.cols + 1):params$data.pack.cols])
wilcox.test(x3[1:params$data.pack.important.cols], x3[(params$data.pack.important.cols + 1):params$data.pack.cols])
wilcox.test(x4[1:params$data.pack.important.cols], x4[(params$data.pack.important.cols + 1):params$data.pack.cols])
wilcox.test(x5[1:params$data.pack.important.cols], x5[(params$data.pack.important.cols + 1):params$data.pack.cols])
