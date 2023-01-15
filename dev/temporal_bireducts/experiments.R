data.pack.rows <- 5000

source('synthetic_data.R')
source('temporal_bireducts_shuffling.R')
source('experiments_utils.R')

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision1",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=50,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=100,
    results.file='results/results_dec1_1.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision1",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=50,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=40,
    results.file='results/results_dec1_2.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision1",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=20,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=100,
    results.file='results/results_dec1_3.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision1",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=20,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=40,
    results.file='results/results_dec1_4.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision2",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=50,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=100,
    results.file='results/results_dec1_5.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision2",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=50,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=40,
    results.file='results/results_dec1_6.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision2",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=20,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=100,
    results.file='results/results_dec1_7.RData'
)

###############################################################################
experiment(
    data.table.generator.function.name="GenerateDataDecision2",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=20,
    bireducts.count=10000,
    bireduct.size.limit=1000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=40,
    results.file='results/results_dec1_8.RData'
)








experiment(
    data.table.generator.function.name="GenerateDataDecision1",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=20,
    bireducts.count=5000,
    bireduct.size.limit=5000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=500,
    results.file='results/results_dec1_500.RData'
)


experiment(
    data.table.generator.function.name="GenerateDataDecision1",
    data.pack.rows=data.pack.rows,
    data.pack.cols=100,
    data.pack.important.cols=20,
    bireducts.count=10000,
    bireduct.size.limit=2000,
    bireduct.size.emptying=100,
    bireduct.desired.add.series.length=300,
    results.file='results/results_dec1_501.RData'
)
