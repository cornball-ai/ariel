% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_cache_load_ptx}
\alias{gpu_cache_load_ptx}
\title{Load Pre-compiled PTX into GPU Kernel Cache}
\usage{
gpu_cache_load_ptx(ptx, kernel_name, shared_mem = 0L)
}
\arguments{
\item{ptx}{Character, PTX assembly text.}

\item{kernel_name}{Character, the kernel entry point name.}

\item{shared_mem}{Integer, shared memory bytes required.}
}
\value{
Invisibly returns a list with ptx, kernel_name, shared_mem.
}
\description{
Loads a PTX string directly into ariel's in-memory kernel cache,
bypassing compilation. Used by artifact loaders to restore compiled
kernels from disk.
}
