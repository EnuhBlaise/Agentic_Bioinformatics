#!/usr/bin/env nextflow
// Cloud variant of the same workflow. Run with -profile awsbatch or -profile k8s (see nextflow.config).
nextflow.enable.dsl = 2
params.samplesheet = null
params.genome_dir = null
params.annotation = null
params.strandedness = 0
params.outdir = 'results'

process FASTP {
    tag "$sample"
    conda 'bioconda::fastp=0.23.4'
    cpus 4
    input: tuple val(sample), path(r1), path(r2)
    output:
      tuple val(sample), path("${sample}_R1.fq.gz"), path("${sample}_R2.fq.gz"), emit: reads
      path "${sample}.fastp.json", emit: json
    publishDir "${params.outdir}/qc", pattern: '*.json', mode: 'copy'
    script:
    """
    fastp -i $r1 -I $r2 -o ${sample}_R1.fq.gz -O ${sample}_R2.fq.gz -j ${sample}.fastp.json -h /dev/null -w $task.cpus
    """
}

process STAR_ALIGN {
    tag "$sample"
    conda 'bioconda::star=2.7.11b'
    cpus 8
    memory '32 GB'
    input:
      tuple val(sample), path(r1), path(r2)
      path genome_dir
    output: path "${sample}.bam", emit: bam
    script:
    """
    STAR --runThreadN $task.cpus --genomeDir $genome_dir --readFilesIn $r1 $r2 --readFilesCommand zcat \\
         --outSAMtype BAM SortedByCoordinate --outFileNamePrefix ${sample}.
    mv ${sample}.Aligned.sortedByCoord.out.bam ${sample}.bam
    """
}

process FEATURECOUNTS {
    conda 'bioconda::subread=2.0.6 conda-forge::python=3.11'
    cpus 8
    publishDir params.outdir, mode: 'copy'
    input:
      path bams
      path annotation
    output: path 'counts.tsv'
    script:
    """
    featureCounts -T $task.cpus -s ${params.strandedness} -p --countReadPairs -a $annotation -o fc.txt $bams
    python3 - <<'PY'
import csv
rows = [r for r in csv.reader(open('fc.txt'), delimiter='\\t') if not r[0].startswith('#')]
hdr = ['gene'] + [c.replace('.bam', '') for c in rows[0][6:]]
with open('counts.tsv', 'w') as fh:
    fh.write('\\t'.join(hdr) + '\\n')
    for r in rows[1:]:
        fh.write('\\t'.join([r[0]] + r[6:]) + '\\n')
PY
    """
}

workflow {
    reads = Channel.fromPath(params.samplesheet).splitCsv(header: true, sep: '\t')
                   .map { row -> tuple(row.sample, file(row.r1), file(row.r2)) }
    FASTP(reads)
    STAR_ALIGN(FASTP.out.reads, file(params.genome_dir))
    FEATURECOUNTS(STAR_ALIGN.out.bam.collect(), file(params.annotation))
}
