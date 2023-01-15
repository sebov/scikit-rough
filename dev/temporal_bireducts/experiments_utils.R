experiment <- function(
    data.table.generator.function.name,
    data.pack.rows,
    data.pack.cols,
    data.pack.important.cols,
    bireducts.count,
    bireduct.size.limit,
    bireduct.size.emptying,
    bireduct.desired.add.series.length,
    description,
    results.file
    ) {

  data.table.generator = eval(parse(text=data.table.generator.function.name))

  result <-  list()
  DS <-  TemporalBireductsDataStream$new()
  rows.processed <- 0

  while (is.null(result$add.series) || length(result$add.series) < bireducts.count) {
    x <- data.table.generator(rows=data.pack.rows, cols=data.pack.cols, important.cols=data.pack.important.cols)
    rows.processed <- rows.processed + nrow(x)
    res =  DS$ProcessDataStreamSamples(
                  x,
                  bireduct.size.limit=bireduct.size.limit,
                  bireduct.size.emptying=bireduct.size.emptying,
                  bireduct.desired.add.series.length=bireduct.desired.add.series.length
                  )
    ## res =  DS$ProcessDataStreamSamples(x, bireduct.size.limit=bireduct.size.limit, bireduct.size.emptying=bireduct.size.emptying, bireduct.desired.add.series.length=bireduct.desired.add.series.length)
    cat("\ncurrent iteration: ", length(res$add.series), "\n", sep="")
    result$add.series <- append(result$add.series, res$add.series)
    result$limit.reached <- append(result$limit.reached, res$limit.reached)
    cat("overall: ", length(result$add.series), "/", bireducts.count, "\n\n", sep="")
  }

  params = list(
      data.table.generator.function.name=data.table.generator.function.name,
      data.pack.rows=data.pack.rows,
      data.pack.cols=data.pack.cols,
      data.pack.important.cols=data.pack.important.cols,
      bireducts.count=bireducts.count,
      bireduct.size.limit=bireduct.size.limit,
      bireduct.size.emptying=bireduct.size.emptying,
      bireduct.desired.add.series.length=bireduct.desired.add.series.length,
      rows.processed=rows.processed
      )
  save(result, params, file=results.file)
}
