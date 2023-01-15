# build a bireduct
CreateTemporalBireduct <- function(stream, cols, rows) {
  real.cols <- stream$current.table.cols.permutation[cols]
  ordered.real.cols <- sort(real.cols)
  real.cols.names <- colnames(stream$data.structure)[real.cols]
  ordered.real.cols.names <- colnames(stream$data.structure)[ordered.real.cols]
  return(list(
          current.table.cols.permutation=stream$current.table.cols.permutation,
          computed.cols=cols,
          real.cols=real.cols,
          real.cols.names=real.cols.names,
          ordered.real.cols=ordered.real.cols,
          ordered.real.cols.names=ordered.real.cols.names,
          rows=rows))
}

# the last attribute is considered decision attribute
CheckFunctionalDependency <- function(table, attr.indices) {
  stopifnot(!(ncol(table) %in% attr.indices))

  if (length(attr.indices) == 0)
    duplicated.without.decision <- nrow(table) - 1
  else
    duplicated.without.decision <- sum(duplicated(table[c(attr.indices)]))

  duplicated.with.decision <- sum(duplicated(table[c(attr.indices, ncol(table))]))

  return(duplicated.without.decision == duplicated.with.decision )
}

# class inside fields and structures
TemporalBireductsDataStream <- setRefClass(
  Class="TemporalBireductsDataStream",
  fields=list(
      current.table.cols.permutation="numeric",
      data.structure="data.frame",
      first.row.index="numeric",
      last.row.index="numeric",
      cols="numeric",
      working.table="data.frame"
  )
)


# it is assumed that new.samples contains attributes in the right order, consistent with
# those stored in state
TemporalBireductsDataStream$methods(
    ProcessDataStreamSamples = function(new.samples, bireduct.size.limit=0, bireduct.size.emptying=1, bireduct.desired.add.series.length) {
      stopifnot(nrow(new.samples) > 0)
      stopifnot((ncol(.self$data.structure) == 0) || (all(colnames(.self$data.structure) == colnames(new.samples))))
      stopifnot(bireduct.size.limit >= 0)
      stopifnot(bireduct.size.limit == 0 || (1 <= bireduct.size.emptying && bireduct.size.emptying <= bireduct.size.limit))
      stopifnot(bireduct.desired.add.series.length > 0)

      result = list(add.series=list(), limit.reached=list())
      new.samples.index <- 0
      if (ncol(.self$data.structure) == 0) {
        .self$data.structure <- data.frame(new.samples[1, ])
        .self$current.table.cols.permutation <- 1:(ncol(.self$data.structure) - 1)
        .self$first.row.index <- 0
        .self$last.row.index <- 0
        .self$cols = numeric()
      }

      # adjust input data to the current columns permutation
      new.samples <- new.samples[c(.self$current.table.cols.permutation, ncol(.self$data.structure))]

      new.samples.index <- 1
      add.series.length <- 0

      nrow.new.samples <- nrow(new.samples)

      while (new.samples.index <= nrow(new.samples)) {
        if (new.samples.index %% 10 == 0) {
          cat("\rsamples iteration: ", format(new.samples.index, width=nchar(nrow.new.samples)), "/", nrow.new.samples, sep="")
        }

        # the limit for the bireduct size reached
        # - remove some of the oldest objects
        # - try to remove unnecessary attributes, trying from the last to the first of the already chosen columns
        if ((bireduct.size.limit > 0) && (nrow(.self$working.table) + 1 > bireduct.size.limit)) {
          # compute the number of objects to remove

          result$limit.reached[[length(result$limit.reached) + 1]] <- CreateTemporalBireduct(stream=.self, cols=.self$cols, rows=c(.self$first.row.index, .self$last.row.index))

          objects.to.remove <- nrow(.self$working.table) - (bireduct.size.limit - bireduct.size.emptying)
          .self$working.table <- .self$working.table[-(1:objects.to.remove), ]
          .self$first.row.index <- .self$first.row.index + objects.to.remove
          col.set.changed <- FALSE
          for (col in rev(.self$cols)) {
            testing.cols <- .self$cols[.self$cols != col]
            if (CheckFunctionalDependency(.self$working.table, testing.cols)) {
              .self$cols = testing.cols
              col.set.changed <- TRUE
            }
          }
          if (col.set.changed == TRUE) {
            add.series.length <- 0
          }
        }

        # add a new record to the end of the table
        if (nrow(.self$working.table) == 0) {
          # shuffle the data
          permutation <- sample(1:(ncol(.self$data.structure) - 1))
          .self$current.table.cols.permutation <- .self$current.table.cols.permutation[permutation]
          new.samples <- new.samples[c(permutation, ncol(.self$data.structure))]
          .self$working.table <- data.frame(new.samples[new.samples.index, ])
          .self$first.row.index <- new.samples.index
          .self$last.row.index <- new.samples.index
          .self$cols = numeric()
          add.series.length <- 0
        } else {
          .self$working.table <- rbind(.self$working.table, new.samples[new.samples.index, ])
          .self$last.row.index <- .self$last.row.index + 1
        }
        new.samples.index <- new.samples.index + 1
        add.series.length <- add.series.length + 1


        if (CheckFunctionalDependency(.self$working.table, .self$cols)) {
          if (add.series.length >= bireduct.desired.add.series.length) {
            # success - save the bireduct of the wanted property
            result$add.series[[length(result$add.series) + 1]] <- CreateTemporalBireduct(stream=.self, cols=.self$cols, rows=c(.self$first.row.index, .self$last.row.index))
            .self$working.table <- data.frame()
          }
        } else {
          # if the functional dependency does not hold
          # - try to add more attributes, from the first to the last
          # - if it is insufficient, remove some of the oldest records
          # - try to remove unnecessary attributes

          col.index <- 1
          while ((col.index < ncol(.self$working.table)) && (! CheckFunctionalDependency(.self$working.table, .self$cols))) {
            if (! col.index %in% .self$cols) {
              .self$cols[[length(.self$cols) + 1]] <- col.index
            }
            col.index <- col.index + 1
          }

          .self$cols <- sort(.self$cols)

          # if attributes are insufficient to discern all objects then remove some of the oldest ones
          while (! CheckFunctionalDependency(.self$working.table, .self$cols)) {
            .self$working.table <- .self$working.table[-1, ]
            .self$first.row.index <- .self$first.row.index + 1
          }

          for (col in rev(.self$cols)) {
            testing.cols <- .self$cols[.self$cols != col]
            if (CheckFunctionalDependency(.self$working.table, testing.cols)) {
              .self$cols = testing.cols
            }
          }

          add.series.length <- 1
        }

      }

      return(result)
    }
)


## TemporalBireductsDataStream$methods(
##     GetPendingBireduct = function() {
##       result = list(CreateTemporalBireduct(table=.self$working.table, cols=.self$cols, rows=c(.self$first.row.index, .self$last.row.index)))
##       return(result)
##     }
## )


## golf <- read.table("golf.csv", header=TRUE)
## DS = TemporalBireductsDataStream$new()
## DS$ProcessDataStreamSamples(golf, desirable.add.series.length=6)


## DS = TemporalBireductsDataStream$new()
## DS$ProcessDataStreamSamples(data.tab.dec.1, bireduct.size.limit=200, bireduct.size.emptying=100, desired.add.series.length=40)
