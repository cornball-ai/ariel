% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_cache_clear}
\alias{gpu_cache_clear}
\title{Clear the GPU Kernel Cache}
\usage{
gpu_cache_clear()
}
\value{
List with n_cleared (number of cached kernels removed)
}
\description{
Unloads all cached PTX modules and clears the kernel cache.
Call this when you want to free GPU resources or reload modified kernels.
}
