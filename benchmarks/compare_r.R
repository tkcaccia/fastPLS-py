args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[[1L]]
output_dir <- args[[2L]]
.libPaths(c(args[[3L]], .libPaths()))
repetitions <- as.integer(args[[4L]])
precision <- args[[5L]]

library(fastPLS)
writeLines(as.character(packageVersion("fastPLS")), file.path(output_dir, "r_version.txt"))

prediction_at <- function(object, index = 1L) {
    value <- object$Ypred
    if (is.list(value)) {
        return(value[[index]])
    }
    if (length(dim(value)) == 3L) {
        return(value[, , index, drop = FALSE][, , 1L])
    }
    value
}

Xtrain <- as.matrix(read.csv(file.path(input_dir, "Xtrain.csv"), header = FALSE))
Xtest <- as.matrix(read.csv(file.path(input_dir, "Xtest.csv"), header = FALSE))
Ytrain <- as.matrix(read.csv(file.path(input_dir, "Ytrain.csv"), header = FALSE))
labels <- factor(readLines(file.path(input_dir, "labels.txt")))
if (identical(precision, "float32")) {
    Xtrain <- float::fl(Xtrain)
    Xtest <- float::fl(Xtest)
    Ytrain <- float::fl(Ytrain)
}

for (method in c("simpls", "plssvd", "opls", "kernelpls")) {
    elapsed <- numeric(repetitions)
    for (iteration in seq_len(repetitions)) {
        started <- proc.time()[["elapsed"]]
        regression <- pls(
            Xtrain, Ytrain, Xtest,
            ncomp = 8,
            method = method,
            backend = "cpu",
            scaling = "centering",
            return_variance = FALSE,
            north = 1L,
            kernel = "rbf",
            gamma = 0.1,
            oversample = 32L,
            power = 5L,
            seed = 17L
        )
        elapsed[[iteration]] <- proc.time()[["elapsed"]] - started
    }
    write.csv(
        as.matrix(prediction_at(regression)),
        file.path(output_dir, paste0("r_", method, "_regression.csv")),
        row.names = FALSE
    )
    writeLines(
        format(median(elapsed), digits = 17),
        file.path(output_dir, paste0("r_", method, "_seconds.txt"))
    )

    classification <- pls(
        Xtrain, labels, Xtest,
        ncomp = 8,
        method = method,
        classifier = "lda",
        backend = "cpu",
        scaling = "centering",
        return_variance = FALSE,
        north = 1L,
        kernel = "rbf",
        gamma = 0.1,
        oversample = 32L,
        power = 5L,
        seed = 17L
    )
    writeLines(
        as.character(prediction_at(classification)),
        file.path(output_dir, paste0("r_", method, "_labels.txt"))
    )
}
