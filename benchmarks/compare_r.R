args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[[1L]]
output_dir <- args[[2L]]
.libPaths(c(args[[3L]], .libPaths()))
repetitions <- as.integer(args[[4L]])

library(fastPLS)

Xtrain <- as.matrix(read.csv(file.path(input_dir, "Xtrain.csv"), header = FALSE))
Xtest <- as.matrix(read.csv(file.path(input_dir, "Xtest.csv"), header = FALSE))
Ytrain <- as.matrix(read.csv(file.path(input_dir, "Ytrain.csv"), header = FALSE))
labels <- factor(readLines(file.path(input_dir, "labels.txt")))

for (method in c("simpls", "plssvd")) {
    elapsed <- numeric(repetitions)
    for (iteration in seq_len(repetitions)) {
        started <- proc.time()[["elapsed"]]
        regression <- pls(
            Xtrain, Ytrain, Xtest,
            ncomp = 8,
            method = method,
            backend = "cpu",
            scaling = "autoscaling",
            return_variance = FALSE,
            rsvd_oversample = 32L,
            rsvd_power = 5L,
            seed = 17L
        )
        elapsed[[iteration]] <- proc.time()[["elapsed"]] - started
    }
    write.csv(
        drop(regression$Ypred),
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
        scaling = "autoscaling",
        return_variance = FALSE,
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        seed = 17L
    )
    writeLines(
        as.character(classification$Ypred[[1L]]),
        file.path(output_dir, paste0("r_", method, "_labels.txt"))
    )
}
