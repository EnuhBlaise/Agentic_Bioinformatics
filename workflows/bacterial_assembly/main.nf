#!/usr/bin/env nextflow
nextflow.enable.dsl = 2
params.reads_r1 = null
params.reads_r2 = null
params.memory_gb = 16
params.outdir = 'results'

process FASTP {
    conda 'bioconda::fastp=0.23.4'
    cpus 4
    publishDir "${params.outdir}/qc", pattern: '*.json', mode: 'copy'
    input: tuple path(r1), path(r2)
    output:
      tuple path('R1.fq.gz'), path('R2.fq.gz'), emit: reads
      path 'fastp.json'
    script: "fastp -i $r1 -I $r2 -o R1.fq.gz -O R2.fq.gz -j fastp.json -h /dev/null -w $task.cpus"
}

process SPADES {
    conda 'bioconda::spades=3.15.5'
    cpus 8
    memory "${params.memory_gb} GB"
    publishDir "${params.outdir}/assembly", mode: 'copy'
    input: tuple path(r1), path(r2)
    output: path 'contigs.fasta'
    script: "spades.py --isolate -1 $r1 -2 $r2 -t $task.cpus -m ${params.memory_gb} -o spades && cp spades/contigs.fasta ."
}

process QUAST {
    conda 'bioconda::quast=5.2.0'
    cpus 4
    publishDir "${params.outdir}", mode: 'copy'
    input: path contigs
    output: path 'quast/report.tsv'
    script: "quast.py $contigs -o quast -t $task.cpus"
}

workflow {
    FASTP(Channel.of(tuple(file(params.reads_r1), file(params.reads_r2))))
    SPADES(FASTP.out.reads)
    QUAST(SPADES.out)
}
