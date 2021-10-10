rm(list = ls())
set.seed(20180912)

# Clean temp dir
temp_dir <- "D:\\UNCC\\EmoDetct\\temp\\"
delfiles <- dir(path=temp_dir, pattern="compare_results*")
file.remove(file.path(temp_dir, delfiles))

# Set working directory
my_wd <- "D:\\UNCC\\EmoDetct"
setwd(paste(my_wd, "\\novels_wasserstein_distance", sep = ''))

for (lib in c('data.table', 'stats', 'Hmisc', 'TDA', 'randomForest', 'caret', 'weights')) {
  if (!lib %in% installed.packages()[, 1]) {
    install.packages(lib)
  }
}

rm(lib)
library(weights)
library(data.table)
library(stats)
library(Hmisc)
# detach("package:transport", unload = TRUE)
library(TDA)
library(caret)

files <- list.files(path = ".")
files <- files[grep('.csv', files)]

writers <- NA
for (i in 1:length(files)) {
  writers[i] <- strsplit(files[i], '\\.')[[1]][1]
}

writers <- as.data.frame(writers)
writers$freq <- 1
writers <- aggregate(freq ~ writers, data = writers, FUN = sum)
writers$description <- paste(writers$writers, ' (', writers$freq / 3, ')', sep = '')

Accuracy.Matrix <- matrix(0, nrow(writers), nrow(writers))
colnames(Accuracy.Matrix) <- writers$description
rownames(Accuracy.Matrix) <- writers$description

# print(writers)
# print(Accuracy.Matrix)
# flush.console()

# Function to get the accuracy of binary classification between each pair of writers
accuracy.function <- function(writer1, writer2, random_name) {

  setwd(paste(my_wd, "\\novels_wasserstein_distance", sep = ''))

  files.selected <- list.files(path = ".")
  files.1 <- files.selected[grepl(writer1, files.selected)]
  files.2 <- files.selected[grepl(writer2, files.selected)]

  names1 <- NA
  names2 <- NA

  for (i in 1:length(files.1)) {
    names1[i] <- strsplit(files.1[i], '\\.')[[1]][2]
  }

  for (i in 1:length(files.2)) {
    names2[i] <- strsplit(files.2[i], '\\.')[[1]][2]
  }

  names1 <- unique(names1)
  names2 <- unique(names2)
  
  if (length(names1) > length(names2)) {
    rand <- sample(1:length(names1), length(names2), replace = F)
    names1 <- names1[rand]
  }

  if (length(names1) < length(names2)) {
    rand <- sample(1:length(names2), length(names1), replace = F)
    names2 <- names2[rand]
  }

  for (i in length(files.1):1) {
    if (!strsplit(files.1[i], '\\.')[[1]][2] %in% names1) {
      files.1 <- files.1[-i]
    }
  }

  for (i in length(files.2):1) {
    if (!strsplit(files.2[i], '\\.')[[1]][2] %in% names2) {
      files.2 <- files.2[-i]
    }
  }

  files.selected <- c(files.1, files.2)

  # load and calculate all persistent diagram based on t=0
  files <- files.selected
  files <- files[grepl("t_zero.csv", files)]
  files <- sort(files)

  novelsRd_t.0 <- list() # A list of Novels in R^d
  novel.writer <- matrix('', nrow = 1, ncol = length(files))
  novel.names <- matrix('', nrow = 1, ncol = length(files))

  for (i in 1:length(files)) {
    f <- files[i]
    novelsRd_t.0[[i]] <- read.csv(f)
    novel.writer[i] <- strsplit(f, '\\.')[[1]][1]
    novel.names[i] <- strsplit(f, '\\.')[[1]][2]
  }

  unique(as.character(novel.writer))
  maxscale = 20 # limit of the filtration
  maxdimension = 2 # components and loops
  novelsPD_t.0 <- list() # A list of Novels Persistent Diagrams

  for (i in 1:(length(files))) {
    novelsPD_t.0[[i]] <- ripsDiag(
      X = novelsRd_t.0[[i]],
      maxdimension = maxdimension,
      maxscale = maxscale,
      dist = "arbitrary",
      library = "Dionysus",
      printProgress = FALSE
    )$diagram
  }

  # load and calculate all persistent diagram based on t=0.1
  files <- files.selected
  files <- files[grepl("t_plus0.1.csv", files)]
  files <- sort(files)
  novelsRd_t.plus <- list() # A list of Novels in R^d

  for (i in 1:(length(files))) {
    f <- files[i]
    novelsRd_t.plus[[i]] <- read.csv(f)
  }

  novelsPD_t.plus <- list() # A list of Novels Persistent Diagrams

  for (i in 1:(length(files))) {
    novelsPD_t.plus[[i]] <- ripsDiag(
      X = novelsRd_t.plus[[i]],
      maxdimension,
      maxscale,
      dist = "arbitrary",
      library = "Dionysus",
      printProgress = FALSE
    )$diagram
  }

  # load and calculate all persistent diagram based on t=-0.1
  files <- files.selected
  files <- files[grepl("t_plus0.1.csv", files)]
  files <- sort(files)
  novelsRd_t.minus <- list() # A list of Novels in R^d

  for (i in 1:(length(files))) {
    f <- files[i]
    novelsRd_t.minus[[i]] <- read.csv(f)
  }

  novelsPD_t.minus <- list() # A list of Novels Persistent Diagrams
  for (i in 1:(length(files))) {
    novelsPD_t.minus[[i]] <- ripsDiag(
      X = novelsRd_t.minus[[i]],
      maxdimension,
      maxscale,
      dist = "arbitrary",
      library = "Dionysus",
      printProgress = FALSE
    )$diagram
  }

  # Now for novels we have: novel.writer, novel.names
  # Persistent diagrams: novelsPD_t.0, novelsPD_t.plus, novelsPD_t.minus
  # We will calculate the matrix of Wasserstein Distances (3 versions for t=0, 0.1 and -0.1)

  W_zero <- matrix(0, nrow = length(files), ncol = length(files))
  W_plus <- matrix(0, nrow = length(files), ncol = length(files))
  W_minus <- matrix(0, nrow = length(files), ncol = length(files))

  for (i in 1:(length(files))) {
    for (j in 1:(length(files))) {
      if (i < j) {
        W_zero[i, j] <- wasserstein(novelsPD_t.0[[i]],
                                    novelsPD_t.0[[j]],
                                    p = 1,
                                    dimension = 1) +
                        wasserstein(novelsPD_t.0[[i]],
                                    novelsPD_t.0[[j]],
                                    p = 1,
                                    dimension = 0)

        W_plus[i, j] <- wasserstein(novelsPD_t.plus[[i]],
                                    novelsPD_t.plus[[j]],
                                    p = 1,
                                    dimension = 1) +
                        wasserstein(novelsPD_t.plus[[i]],
                                    novelsPD_t.plus[[j]],
                                    p = 1,
                                    dimension = 0)

        W_minus[i, j] <- wasserstein(novelsPD_t.minus[[i]],
                                     novelsPD_t.minus[[j]],
                                     p = 1,
                                     dimension = 1) +
                          wasserstein(novelsPD_t.minus[[i]],
                                      novelsPD_t.minus[[j]],
                                      p = 1,
                                      dimension = 1)
      }
      if (i > j) {
        W_zero[i, j] <- W_zero[j, i]
        W_plus[i, j] <- W_plus[j, i]
        W_minus[i, j] <- W_minus[j, i]
      }
    }
  }

  # Prepare folds for cross validation
  fold <- sample(1:length(files), length(files), replace = F)
  folds <- list()
  folds[[1]] <- fold[1:round(0.5 * length(files))]
  folds[[2]] <- fold[(1 + round(0.5 * length(files))):round(1.0 * length(files))]

  Trains1 <- list() # sequence of 10 train data
  Trains3 <- list() # sequence of 10 train data baed on 3 measure of distances

  Tests1 <- list() # sequence of 10 test data
  Tests3 <- list() # sequence of 10 test data baed on 3 measure of distances

  # k-NN
  TestsKNN <- list() # a list to keep KNN test sets

  for (i in 1:length(folds)) {
    fold <- folds[[i]]
    testKNN <- as.data.frame(novel.names[fold])
    names(testKNN) <- 'novel'
    testKNN$writer <- novel.writer[fold]
    testKNN$prediction <- NA
    for (ind in 1:(length(fold))) {
      x <- fold[ind]
      # Column of distances
      temp1 <- sqrt((W_zero[-fold, x]) ^ 2 + (W_plus[-fold, x]) ^ 2 + (W_minus[-fold, x]) ^ 2)
      temp2 <- novel.writer[-fold] #column of labels
      temp <- data.frame(distance = temp1, writer = temp2)
      temp <- temp[order(temp$distance, decreasing = F),]
      N <- 5
      temp <- temp[1:N,] # 5-N-N
      temp$freq <- (temp$distance) ^ (-2)
      temp <- aggregate(freq ~ writer, data = temp, FUN = sum)
      temp <- temp[order(temp$freq, decreasing = T),]
      testKNN$writer[ind]
      testKNN$prediction[ind] <- as.character(temp$writer[1])
    }
    TestsKNN[[i]] <- testKNN
    rm(testKNN)
  }

  # Now, Prediction and material for confusion matrix are Tests KNN
  results <- TestsKNN[[1]]
  for (i in 2:length(folds)) {
    results <- rbind(results, TestsKNN[[i]])
  }

  # typeof(results) => list
  cm <- confusionMatrix(as.factor(results$prediction), as.factor(results$writer))
  random2 <- runif(1, 1, 2000)
  rand_name <- paste('compare_results', random_name, random2, '.csv', sep = '')
  setwd(paste(my_wd, "\\temp", sep = ''))
  fwrite(results, rand_name)
  setwd(paste(my_wd, "\\novels_wasserstein_distance", sep = ''))

  # End of the distance function
  return(cm$overall['Accuracy'])
}

# Results Comparison
random_name <- 100000 + sample(1:100000, 1)
for (i in 1:(nrow(writers) - 1)) {
  for (j in (i + 1):nrow(writers)) {
    T = 250
    x <- NA
    for (times in 1:T) {
      cat(writers$writers[i], file = "output.txt", sep = "\n", append = TRUE)
      cat(writers$writers[j], file = "output.txt", sep = "\n", append = TRUE)
      x[times] <- accuracy.function(writers$writers[i],
                                    writers$writers[j],
                                    random_name)
    }
    x <- mean(x)
    Accuracy.Matrix[i, j] <- x
    Accuracy.Matrix[j, i] <- x
  }
}

setwd(paste(my_wd, "\\temp", sep = ''))
temp.files <- list.files(path = ".")
temp.files <- temp.files[grepl(as.character(random_name), temp.files)]
results <- data.frame(matrix(nrow = 0, ncol = 3))
names(results) <- c('novel', 'writer', 'prediction')
for (i in 1:length(temp.files)) {
  temp <- fread(temp.files[i])
  results <- rbind(results, temp)
}
setwd(paste(my_wd, "\\novels_ph_results", sep = ''))
fwrite(results, 'results_comparison_wasserstein.csv')

# Results Summary
results$count <- 1
results1 <- aggregate(count ~ novel, data = results, FUN = sum)
results$count <- NULL
results$correct <- ifelse(results$writer == results$prediction, 1, 0)
results2 <- aggregate(correct ~ ., data = results[, - 'prediction'], FUN = sum)
results.summary <- merge(results1, results2, by = 'novel')
results.summary <- results.summary[, c('novel', 'writer', 'count', 'correct')]
results.summary$Accuracy <- results.summary$correct / results.summary$count
results.summary <- results.summary[order(results.summary$Accuracy, results.summary$writer),]
fwrite(results.summary, 'results_summary_wasserstein.csv')

# Accuracy Matrix
diag(Accuracy.Matrix) <- NA
Accuracy.Matrix <- round(100 * Accuracy.Matrix, 1)
diag(Accuracy.Matrix) <- NA
out <- cbind(rownames(Accuracy.Matrix), as.data.frame(Accuracy.Matrix))
out$Average <- rowMeans(out[, 2:(dim(out)[2])], na.rm = T)
names(out)[1] <- NA
fwrite(out, 'accuracy_matrix_wasserstein.csv')
diag(Accuracy.Matrix) <- NA
print(mean(Accuracy.Matrix, na.rm = T))
Accuracy.Matrix

# Clean temp dir
temp_dir <- "D:\\UNCC\\EmoDetct\\temp\\"
delfiles <- dir(path=temp_dir, pattern="compare_results*")
file.remove(file.path(temp_dir, delfiles))