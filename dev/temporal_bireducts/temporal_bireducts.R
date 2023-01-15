# build a bireduct
CreateTemporalBireduct <- function(table, cols, rows) {
  return(list(col.names=colnames(table)[cols], cols=cols, rows=rows))
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
      first.row.index="numeric",
      last.row.index="numeric",
      cols="numeric",
      table="data.frame"
  )
)


# it is assumed that new.samples contains attributes in the right order, consistent with
# those stored in state
TemporalBireductsDataStream$methods(
    ProcessDataStreamSamples = function(new.samples, bireduct.size.limit=0, bireduct.size.emptying=1) {
      stopifnot(nrow(new.samples) > 0)
      stopifnot((ncol(.self$table) == 0) || (all(colnames(.self$table) == colnames(new.samples))))
      stopifnot(bireduct.size.limit >= 0)
      stopifnot(bireduct.size.limit == 0 || (1 <= bireduct.size.emptying && bireduct.size.emptying <= bireduct.size.limit))

      result = list()
      new.samples.index <- 1
      if (ncol(.self$table) == 0) {
        .self$table <- data.frame(new.samples[new.samples.index, ])
        new.samples.index <- new.samples.index + 1
        .self$first.row.index <- 1
        .self$last.row.index <- 1
        .self$cols = numeric()
      }

      while (new.samples.index <= nrow(new.samples)) {
        if (new.samples.index %% 10 == 0) {
          print(new.samples.index)
        }
        oversized.bireduct.saved <- FALSE

        # the limit for the bireduct size reached
        # - remove some of the oldest objects
        # - try to remove unnecessary attributes, trying from the last to the first of the already chosen columns
        if ((bireduct.size.limit > 0) && (nrow(.self$table) + 1 > bireduct.size.limit)) {
          result[[length(result) + 1]] <- CreateTemporalBireduct(table=.self$table, cols=.self$cols, rows=c(.self$first.row.index, .self$last.row.index))
          oversized.bireduct.saved <- TRUE

          # compute the number of objects to remove
          objects.to.remove <- nrow(.self$table) - (bireduct.size.limit - bireduct.size.emptying)

          .self$table <- .self$table[-(1:objects.to.remove), ]
          .self$first.row.index <- .self$first.row.index + objects.to.remove
          for (col in rev(.self$cols)) {
            testing.cols <- .self$cols[.self$cols != col]
            if (CheckFunctionalDependency(.self$table, testing.cols)) {
              .self$cols = testing.cols
            }
          }
        }

        # add a new record to the end of the table
        .self$table <- rbind(.self$table, new.samples[new.samples.index, ])
        new.samples.index <- new.samples.index + 1
        .self$last.row.index <- .self$last.row.index + 1

        # if the functional dependency does not hold
        # - try to add more attributes, from the first to the last
        # - if it is insufficient, remove some of the oldest records
        # - try to remove unnecessary attributes
        if (!CheckFunctionalDependency(.self$table, .self$cols)) {
          if (oversized.bireduct.saved == FALSE) {
            result[[length(result) + 1]] <- CreateTemporalBireduct(table=.self$table, cols=.self$cols, rows=c(.self$first.row.index, .self$last.row.index - 1))
          }

          col.index <- 1
          while ((col.index < ncol(.self$table)) && (! CheckFunctionalDependency(.self$table, .self$cols))) {
            if (! col.index %in% .self$cols) {
              .self$cols[[length(.self$cols) + 1]] <- col.index
            }
            col.index <- col.index + 1
          }

          .self$cols <- sort(.self$cols)

          # if attributes are insufficient to discern all objects then remove some of the oldest ones
          while (! CheckFunctionalDependency(.self$table, .self$cols)) {
            .self$table <- .self$table[-1, ]
            .self$first.row.index <- .self$first.row.index + 1
          }


          for (col in rev(.self$cols)) {
            testing.cols <- .self$cols[.self$cols != col]
            if (CheckFunctionalDependency(.self$table, testing.cols)) {
              .self$cols = testing.cols
            }
          }
        }
      }

      return(result)
    }
)


TemporalBireductsDataStream$methods(
    GetPendingBireduct = function() {
      result = list(CreateTemporalBireduct(table=.self$table, cols=.self$cols, rows=c(.self$first.row.index, .self$last.row.index)))
      return(result)
    }
)


## golf <- read.table("golf.csv", header=TRUE)
## DS = TemporalBireductsDataStream$new()
## DS$ProcessDataStreamSamples(golf[,c(1,3,4,5)], 3, 1)
## DS$GetPendingBireduct()
