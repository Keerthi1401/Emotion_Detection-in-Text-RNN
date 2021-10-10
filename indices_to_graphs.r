rm(list = ls())

my_wd <- "D:\\UNCC\\EmoDetct"
setwd(my_wd)

for (lib in c('data.table', 'stats', 'Hmisc', 'transport')) {
  if (!lib %in% installed.packages()[, 1]) {
    install.packages(lib)
  }
}

rm(lib)
library(data.table)
setwd(paste(my_wd, '\\novels_raw_index', sep = ''))
files <- list.files(path = ".")
head(files)
novels <- list()

for (i in 1:length(files)) {
  novels[[i]] <- fread(files[i])
  names(novels[[i]]) <- c('ind', 'ch')
  novels[[i]]$count <- 1
}

temp <- files
novelists <- files
novel.names <- files

for (i in 1:length(files)) {
  temp[i] <- substr(temp[i], 4, nchar(temp[i]))
  novel.names[i] <- strsplit(temp[i], split = ' by ')[[1]][1]
  novelists[i] <- strsplit(temp[i], split = ' by ')[[1]][2]
  novelists[i] <- strsplit(novelists[i], split = '\\.')[[1]][1]
}
rm(temp)
print(novel.names)
print(novelists)

ch.count <- list()
for (i in 1:length(novels)) {
  novels[[i]]$ch <- tolower(novels[[i]]$ch)
  ch.count[[i]] <- aggregate(count ~ ch, data = novels[[i]], FUN = sum)
  ch.count[[i]] <- ch.count[[i]][order(ch.count[[i]]$count, decreasing = T),]
  ch.count[[i]] <- ch.count[[i]][1:10,]
  novels[[i]]$count <- NULL
  ch.count[[i]] <- ch.count[[i]][!is.na(ch.count[[i]]$count),]
  novels[[i]] <- merge(novels[[i]], ch.count[[i]], by = 'ch')
}
rm(ch.count)

print("Novel Details...")
for (i in 1:length(novels)) {
  print('****************')
  cat(sprintf("Novel: %s", novel.names[i]))
  cat(sprintf("Total: %s", length(novels[[i]]$ch)))
  cat(sprintf("Unique: %s", length(unique(novels[[i]]$ch))))
}

print("Retrieving useless novels...")
useless.list <- list()
for (i in 1:length(novels)) {
  if (length(unique(novels[[i]]$ch)) < 10) {
    print('****************')
    cat(sprintf("Novel: %s", novel.names[i]))
    cat(sprintf("Total: %s", length(novels[[i]]$ch)))
    cat(sprintf("Unique: %s", length(unique(novels[[i]]$ch))))
    useless.list <- append(useless.list, i)
  }
}

# if (length(delete.list) > 0) {
#   print("There are useless index files.")
#   for (i in length(delete.list):1) {
#     for.delete <- delete.list[[i]]
#     novels[[for.delete]] <- NULL
#     novelists <- novelists[-for.delete]
#     novel.names <- novel.names[-for.delete]
#   }
# }

setwd(paste(my_wd, '\\novels_wasserstein_distance', sep = ''))

x <- matrix(0, nrow = 20, ncol = 3)
x <- as.data.frame(x)

library(transport)

# Function to measure two characters in a novel
distance.function <- function(in1, in2, Novel.Length, t) {
  # in1 should be always larger than in2
  in1 <- sort(in1)
  in2 <- sort(in2)
  if (length(in2) > length(in1)) {
    temp <- in1
    in1 <- in2
    in2 <- temp
    rm(temp)
  }
  in1ed <- in2
  y_nearest <- 0
  for (x in 1:length(in2)) {
    used.list <- c()
    in1ed[x] <- Inf
    # for index in in2 finding the nearest index in in1
    temp_dist <- Inf
    current_dist <- 0
    for (y in 1:length(in1)) {
      current_dist <- (in2[x] - in1[y]) ^ 2
      if (length(current_dist) > 0 && !is.na(current_dist)) {
        if (current_dist < temp_dist & (!y %in% used.list)) {
          temp_dist <- current_dist
          y_nearest <- y
        }
        in1ed[x] <- in1[y_nearest]
        used.list <- c(used.list, y_nearest)
      }
    }
  }
  in1ed <- sort(in1ed)
  # Normalizing
  in1ed <- in1ed / Novel.Length
  in2 <- in2 / Novel.Length
  for (x in 1:length(in2)) {
    in1ed[x] <- in1ed[x] ^ (1 + t)
    in2[x] <- in2[x] ^ (1 + t)
  }
  d <- wasserstein1d(in1ed, in2, p = 0.5)
  return(d)
}

for (n in 1:length(novels)) {
  inx <- novels[[n]]
  novel.lenghth <- max(novels[[n]]$ind)
  chars <- unique(inx$ch)
  # Define 3 matrices to reserve distances with 3 parameters 0, -0.1, +0.1
  dist.t_zero <- matrix(0, 10, 10)
  dist.t_plus0.1 <- matrix(0, 10, 10)
  dist.t_minus0.1 <- matrix(0, 10, 10)
  for (i in 1:9) {
    for (j in (i + 1):10) {
      ch1 <- chars[i]
      ch2 <- chars[j]
      slice1 <- inx[inx$ch == ch1,]
      slice2 <- inx[inx$ch == ch2,]
      slice1 <- slice1$ind
      slice2 <- slice2$ind
      # Parameter t = 0
      dist.t_zero[i, j] <- distance.function(slice1, slice2, novel.lenghth, 0)
      dist.t_zero[j, i] <- dist.t_zero[i, j]
      # Parameter t =+ 0.1
      dist.t_plus0.1[i, j] <- distance.function(slice1, slice2, novel.lenghth, 0.1)
      dist.t_plus0.1[j, i] <- dist.t_plus0.1[i, j]
      # Parameter t =- 0.1
      dist.t_minus0.1[i, j] <- distance.function(slice1, slice2, novel.lenghth, -0.1)
      dist.t_minus0.1[j, i] <- dist.t_minus0.1[i, j]
    }
  }
  novelists[n] <- gsub('\\.', '', novelists[n])
  novel.names[n] <- gsub('\\.', '', novel.names[n])
  name1 <- paste(novelists[n], novel.names[n], 't_zero', 'csv', sep = '.')
  name2 <- paste(novelists[n], novel.names[n], 't_plus0.1', 'csv', sep = '.')
  name3 <- paste(novelists[n], novel.names[n], 't_minus0.1', 'csv', sep = '.')
  fwrite(as.data.frame(dist.t_zero), name1)
  fwrite(as.data.frame(dist.t_plus0.1), name2)
  fwrite(as.data.frame(dist.t_minus0.1), name3)
}