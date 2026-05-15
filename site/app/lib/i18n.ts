import type { MetricStatus } from "./format";

export type Locale = "en" | "ko";

export const defaultLocale: Locale = "en";
export const supportedLocales = ["en", "ko"] as const;

export type BasePagePath = "/" | "/accuracy" | "/speed" | "/methodology" | "/data";

const basePagePaths = ["/", "/accuracy", "/speed", "/methodology", "/data"] as const;
const basePagePathSet = new Set<string>(basePagePaths);

export type NavKey = "summary" | "accuracy" | "speed" | "methodology" | "data";

export type KnownCaveatId =
  | "latency-not-measured"
  | "generative-exact-match"
  | "agentic-scaffold-dependent"
  | "diagnostic-sourceqa"
  | "diagnostic-memory-stability"
  | "coverage-missing"
  | "optional-eval-lane"
  | "mtplx-speed-only";

export type Messages = {
  root: {
    brandAria: string;
    brandSubtitle: string;
    languageAria: string;
    languageLabels: Record<Locale, string>;
    navAria: string;
    nav: Record<NavKey, string>;
  };
  common: {
    allFamilies: string;
    allDimensions: string;
    allTasks: string;
    caveatsTracked: string;
    comparisons: string;
    gen: string;
    notAvailable: string;
    of: string;
    prompt: string;
    rows: string;
    scenarios: string;
    unknown: string;
  };
  status: Record<MetricStatus, string>;
  caveats: Record<KnownCaveatId, string>;
  tables: {
    accuracy: {
      aria: string;
      empty: string;
      headers: {
        caveats: string;
        metric: string;
        model: string;
        runId: string;
        score: string;
        taskDim: string;
      };
    };
    coverage: {
      aria: string;
      empty: string;
      headers: {
        dim: string;
        lane: string;
        model: string;
        runner: string;
        status: string;
        task: string;
      };
    };
    speed: {
      aria: string;
      empty: string;
      headers: {
        itl: string;
        memory: string;
        model: string;
        pp: string;
        scenario: string;
        status: string;
        tg: string;
        ttft: string;
        wall: string;
      };
    };
    glossary: {
      aria: string;
      headers: {
        caveat: string;
        interpretation: string;
        metric: string;
        status: string;
      };
      items: Array<{
        caveat: string;
        label: string;
        status: MetricStatus;
        summary: string;
      }>;
    };
    scenario: {
      aria: string;
      headers: {
        generationTokens: string;
        promptTokens: string;
        scenario: string;
        use: string;
      };
      useText: string;
    };
  };
  downloads: Array<{
    description: string;
    href: string;
    label: string;
  }>;
  pages: {
    summary: {
      eyebrow: string;
      titlePrefix: string;
      leadStart: string;
      leadMiddle: string;
      leadEnd: string;
      metadataAria: string;
      meta: {
        accuracyRows: string;
        hardware: string;
        speedRows: string;
        variants: string;
      };
      findingsAria: string;
      findings: {
        topHumanEval: {
          label: string;
          empty: string;
        };
        fastest: {
          label: string;
          empty: string;
          detail: (model: string, memory: string) => string;
        };
        mtplx: {
          label: string;
          empty: string;
          detail: (model: string, scenario: string, acceptance: string) => string;
        };
        caveatCoverage: {
          label: string;
          valueSuffix: string;
          detail: string;
        };
      };
      coverageTitle: string;
      coverageBody: (missing: number, optional: number, speedOnly: number) => string;
      caveatTitle: string;
      caveatBody: string;
      accuracySnapshot: {
        eyebrow: string;
        title: string;
        aside: string;
      };
      throughput: {
        eyebrow: string;
        title: string;
        aside: string;
      };
    };
    accuracy: {
      eyebrow: string;
      title: string;
      lead: string;
      filtersAria: string;
      taskLabel: string;
      dimensionLabel: string;
      familyLabel: string;
      coverageEyebrow: string;
      coverageTitle: string;
      coverageAside: string;
      tableEyebrow: string;
      tableTitle: string;
      sortedBy: string;
    };
    speed: {
      eyebrow: string;
      title: string;
      lead: string;
      filtersAria: string;
      scenarioLabel: string;
      topTgEyebrow: string;
      generationTitle: string;
      topByTok: string;
      latencyTitle: string;
      latencyBodyStart: string;
      latencyBodyEnd: (benchVersion: string) => string;
      tableEyebrow: string;
      tableTitle: string;
      sortedBy: string;
      mtplxEyebrow: string;
      mtplxTitle: string;
    };
    methodology: {
      eyebrow: string;
      title: string;
      lead: string;
      contextAria: string;
      readingEyebrow: string;
      comparisonRulesTitle: string;
      rules: string[];
      metricsEyebrow: string;
      metricGlossaryTitle: string;
      latencyEyebrow: string;
      latencyTitle: string;
      latencyBody: (benchVersion: string) => string;
      accuracyEyebrow: string;
      accuracyTitle: string;
      accuracyBody: string;
      scenariosEyebrow: string;
      scenarioTitle: string;
    };
    data: {
      eyebrow: string;
      title: string;
      lead: string;
      contextAria: string;
      rowsLabel: string;
      rowsValue: (accuracyRows: number, speedRows: number, mtplxRows: number) => string;
      artifactsEyebrow: string;
      downloadsTitle: string;
      servedFrom: string;
    };
  };
  mtplx: {
    empty: string;
    speedupSuffix: string;
    versus: string;
    labels: {
      arTg: string;
      acceptD1: string;
      acceptD2: string;
      acceptD3: string;
      mtpTg: string;
      verify: string;
    };
  };
};

export const messages = {
  en: {
    root: {
      brandAria: "llm-bench summary",
      brandSubtitle: "Apple Silicon local model benchmarks",
      languageAria: "Language",
      languageLabels: {
        en: "EN",
        ko: "KO",
      },
      navAria: "Primary navigation",
      nav: {
        summary: "Summary",
        accuracy: "Accuracy",
        speed: "Speed",
        methodology: "Methodology",
        data: "Data",
      },
    },
    common: {
      allFamilies: "All families",
      allDimensions: "All dimensions",
      allTasks: "All tasks",
      caveatsTracked: "caveats tracked",
      comparisons: "comparisons",
      gen: "gen",
      notAvailable: "not available",
      of: "of",
      prompt: "prompt",
      rows: "rows",
      scenarios: "scenarios",
      unknown: "unknown",
    },
    status: {
      measured: "measured",
      directional: "directional",
      diagnostic: "diagnostic",
      unavailable: "not measured",
      optional: "optional",
      speed_only: "speed-only",
      missing: "missing",
      unsupported: "unsupported",
    },
    caveats: {
      "latency-not-measured":
        "bench_version 0.3 does not directly measure TTFT or ITL.",
      "generative-exact-match":
        "Generative exact-match rows can undercount correct answers because of output formatting and answer extraction.",
      "agentic-scaffold-dependent":
        "ProgramBench scores include the agent scaffold, tools, and execution environment, not only the base model.",
      "diagnostic-sourceqa":
        "SourceQA is a small source-grounding smoke/regression check, so it is not used for headline ranking or primary coverage debt.",
      "diagnostic-memory-stability":
        "Memory Stability is a synthetic diagnostic for repeated memory rewriting, so it is not used for headline ranking or primary coverage debt.",
      "coverage-missing":
        "A primary supported evaluation has no committed result yet.",
      "optional-eval-lane":
        "This benchmark is tracked as an optional lane and does not block the primary matrix.",
      "mtplx-speed-only":
        "MTPLX MTP/AR variants are speedup comparators, so they do not add accuracy coverage debt.",
    },
    tables: {
      accuracy: {
        aria: "Accuracy results",
        empty: "No accuracy rows match the current filters.",
        headers: {
          caveats: "Status / caveats",
          metric: "Metric",
          model: "Model",
          runId: "Run ID",
          score: "Score",
          taskDim: "Task / dim",
        },
      },
      coverage: {
        aria: "Evaluation coverage",
        empty: "No coverage rows match the current filters.",
        headers: {
          dim: "Dimension",
          lane: "Lane",
          model: "Model",
          runner: "Runner",
          status: "Status",
          task: "Task",
        },
      },
      speed: {
        aria: "Speed scenario results",
        empty: "No speed rows match the current scenario.",
        headers: {
          itl: "ITL",
          memory: "Memory",
          model: "Model",
          pp: "PP",
          scenario: "Scenario",
          status: "Status",
          tg: "TG",
          ttft: "TTFT",
          wall: "Wall",
        },
      },
      glossary: {
        aria: "Metric glossary",
        headers: {
          caveat: "Caveat",
          interpretation: "Interpretation",
          metric: "Metric",
          status: "Status",
        },
        items: [
          {
            caveat: "Measured from benchmark run summaries.",
            label: "PP tok/s",
            status: "measured",
            summary: "Prompt processing throughput. Higher means faster prompt ingestion.",
          },
          {
            caveat: "Primary speed comparison metric for this export.",
            label: "TG tok/s",
            status: "measured",
            summary: "Generation throughput during decode. Higher means faster output tokens.",
          },
          {
            caveat: "Reported as peak resident memory across runs.",
            label: "Peak memory",
            status: "measured",
            summary: "Mean peak memory in GB for the benchmark scenario.",
          },
          {
            caveat: "End-to-end scenario duration.",
            label: "Wall time",
            status: "measured",
            summary: "Mean elapsed seconds for the prompt and generation scenario.",
          },
          {
            caveat: "bench_version 0.3 exports null latency fields.",
            label: "TTFT",
            status: "unavailable",
            summary: "Time to first token in milliseconds. Not measured in the current export.",
          },
          {
            caveat: "bench_version 0.3 exports null latency fields.",
            label: "ITL",
            status: "unavailable",
            summary: "Inter-token latency in milliseconds. Not measured in the current export.",
          },
          {
            caveat: "Some generative exact-match rows can undercount valid formatted answers.",
            label: "Accuracy",
            status: "directional",
            summary: "Task score from committed evaluation artifacts. Higher means better task performance.",
          },
          {
            caveat: "MTP-on divided by autoregressive baseline for matching scenario pairs.",
            label: "MTPLX speedup",
            status: "measured",
            summary: "Relative generation speedup for speculative decoding comparisons.",
          },
        ],
      },
      scenario: {
        aria: "Speed scenario matrix",
        headers: {
          generationTokens: "Generation tokens",
          promptTokens: "Prompt tokens",
          scenario: "Scenario",
          use: "Use",
        },
        useText: "Same-scenario speed comparison.",
      },
    },
    downloads: [
      {
        description: "Typed site export used by the report pages.",
        href: "/data/benchmarks.json",
        label: "benchmarks.json",
      },
      {
        description: "Combined benchmark speed summary.",
        href: "/data/summary.csv",
        label: "summary.csv",
      },
      {
        description: "Primary evaluation score summary.",
        href: "/data/eval_summary_primary.csv",
        label: "eval_summary_primary.csv",
      },
      {
        description: "MTPLX speculative decoding speedup rows.",
        href: "/data/mtplx_speedups.csv",
        label: "mtplx_speedups.csv",
      },
      {
        description: "Registry and evaluation coverage status snapshot.",
        href: "/data/index.json",
        label: "index.json",
      },
    ],
    pages: {
      summary: {
        eyebrow: "Benchmark report",
        titlePrefix: "Local LLM results on",
        leadStart: "Static report generated from committed benchmark artifacts at source commit",
        leadMiddle: "Current coverage includes",
        leadEnd:
          "accuracy, prompt/generation throughput, peak memory, wall time, and MTPLX speedups.",
        metadataAria: "Dataset metadata",
        meta: {
          accuracyRows: "Accuracy rows",
          hardware: "Hardware",
          speedRows: "Speed rows",
          variants: "Variants",
        },
        findingsAria: "Key findings",
        findings: {
          topHumanEval: {
            label: "Top HumanEval",
            empty: "No HumanEval rows are present in the current data export.",
          },
          fastest: {
            label: "Fastest p256_g128",
            empty: "No p256_g128 speed rows are present in the current data export.",
            detail: (model, memory) => `${model} at ${memory} peak memory.`,
          },
          mtplx: {
            label: "MTPLX speedup",
            empty: "No MTPLX comparison rows are present in the current data export.",
            detail: (model, scenario, acceptance) =>
              `${model} on ${scenario}, with d1 acceptance ${acceptance}.`,
          },
          caveatCoverage: {
            label: "Caveat coverage",
            valueSuffix: "tracked",
            detail:
              "Latency metrics and generative exact-match interpretation are explicitly marked where applicable.",
          },
        },
        coverageTitle: "Coverage first",
        coverageBody: (missing, optional, speedOnly) =>
          `${missing} primary coverage gaps, ${optional} optional lanes, and ${speedOnly} MTPLX speed-only rows are visible before score ranking.`,
        caveatTitle: "Metric caveat",
        caveatBody:
          "TTFT and ITL are unavailable in this export, so speed comparisons use generation tokens per second. Generative exact-match accuracy rows can be directional when answer extraction or formatting may undercount correct outputs.",
        accuracySnapshot: {
          eyebrow: "Accuracy snapshot",
          title: "Top HumanEval rows",
          aside: "Top 5 by score",
        },
        throughput: {
          eyebrow: "Generation throughput",
          title: "Fastest p256_g128 rows",
          aside: "Top 5 by TG tok/s",
        },
      },
      accuracy: {
        eyebrow: "Accuracy explorer",
        title: "Committed accuracy rows",
        lead:
          "Filter committed benchmark accuracy artifacts by task, dimension, and model family. Coverage is shown before scores so missing tests are not mistaken for weak scores.",
        filtersAria: "Accuracy filters",
        taskLabel: "Task",
        dimensionLabel: "Dimension",
        familyLabel: "Family",
        coverageEyebrow: "Coverage",
        coverageTitle: "Which tests each model has taken",
        coverageAside: "Status from results/index.json",
        tableEyebrow: "Explorer",
        tableTitle: "Accuracy results",
        sortedBy: "Sorted by task, score, model",
      },
      speed: {
        eyebrow: "Speed explorer",
        title: "Token throughput, memory, and MTPLX speedups",
        lead:
          "Filter committed speed artifacts by prompt and generation shape. The report keeps PP, TG, memory, wall time, and MTPLX comparisons dense for benchmark review.",
        filtersAria: "Speed filters",
        scenarioLabel: "Scenario",
        topTgEyebrow: "Top TG",
        generationTitle: "Generation throughput",
        topByTok: "Top 8 by tok/s",
        latencyTitle: "TTFT and ITL",
        latencyBodyStart:
          "TTFT and ITL are intentionally displayed from measured fields only. They remain",
        latencyBodyEnd: (benchVersion) =>
          `where bench_version ${benchVersion} exports null latency values.`,
        tableEyebrow: "Explorer",
        tableTitle: "Speed results",
        sortedBy: "Sorted by TG tok/s",
        mtplxEyebrow: "MTPLX",
        mtplxTitle: "MTP-on versus AR baseline",
      },
      methodology: {
        eyebrow: "Methodology",
        title: "How to read this benchmark report",
        lead:
          "Results are static snapshots from committed benchmark artifacts. Treat scores as local Apple Silicon measurements for the listed hardware, benchmark version, and source commit.",
        contextAria: "Benchmark context",
        readingEyebrow: "Reading order",
        comparisonRulesTitle: "Comparison rules",
        rules: [
          "Compare speed rows within the same scenario only.",
          "Use TG tok/s as the primary throughput metric for generation speed.",
          "Use peak memory and wall time to identify tradeoffs hidden by throughput alone.",
          "Read accuracy by task; benchmarks with different prompts or graders are not pooled.",
          "Prefer measured rows over directional rows when drawing strict conclusions.",
        ],
        metricsEyebrow: "Metrics",
        metricGlossaryTitle: "Metric glossary",
        latencyEyebrow: "Latency caveat",
        latencyTitle: "TTFT and ITL unavailable",
        latencyBody: (benchVersion) =>
          `TTFT and ITL are included as columns because they are useful latency metrics, but the current export records them as null. For bench_version ${benchVersion}, latency comparisons should use TG tok/s, wall time, and memory instead.`,
        accuracyEyebrow: "Accuracy caveat",
        accuracyTitle: "Directional exact-match rows",
        accuracyBody:
          "Some generative tasks use exact-match or extraction-based scoring. Formatting, alternate equivalent answers, and answer extraction can undercount correct responses, so rows marked directional should be read as comparative signals, not final capability claims.",
        scenariosEyebrow: "Speed scenarios",
        scenarioTitle: "Prompt and generation matrix",
      },
      data: {
        eyebrow: "Data",
        title: "Download benchmark artifacts",
        lead:
          "Public files mirror the committed CSV summaries and generated JSON export used by this site.",
        contextAria: "Data export context",
        rowsLabel: "Rows",
        rowsValue: (accuracyRows, speedRows, mtplxRows) =>
          `${accuracyRows} accuracy / ${speedRows} speed / ${mtplxRows} MTPLX`,
        artifactsEyebrow: "Artifacts",
        downloadsTitle: "Static downloads",
        servedFrom: "Served from /data",
      },
    },
    mtplx: {
      empty: "No MTPLX speedup rows are present.",
      speedupSuffix: "speedup",
      versus: "vs",
      labels: {
        arTg: "AR TG",
        acceptD1: "Accept d1",
        acceptD2: "Accept d2",
        acceptD3: "Accept d3",
        mtpTg: "MTP TG",
        verify: "Verify",
      },
    },
  },
  ko: {
    root: {
      brandAria: "llm-bench 요약",
      brandSubtitle: "Apple Silicon 로컬 모델 벤치마크",
      languageAria: "언어",
      languageLabels: {
        en: "EN",
        ko: "KO",
      },
      navAria: "주요 내비게이션",
      nav: {
        summary: "요약",
        accuracy: "정확도",
        speed: "속도",
        methodology: "방법론",
        data: "데이터",
      },
    },
    common: {
      allFamilies: "전체 family",
      allDimensions: "전체 dimension",
      allTasks: "전체 task",
      caveatsTracked: "개 caveat 추적",
      comparisons: "개 비교",
      gen: "gen",
      notAvailable: "없음",
      of: "중",
      prompt: "prompt",
      rows: "개 행",
      scenarios: "개 scenario",
      unknown: "unknown",
    },
    status: {
      measured: "측정됨",
      directional: "방향성",
      diagnostic: "진단용",
      unavailable: "미측정",
      optional: "옵션",
      speed_only: "속도 전용",
      missing: "누락",
      unsupported: "미지원",
    },
    caveats: {
      "latency-not-measured":
        "bench_version 0.3에서는 TTFT와 ITL을 직접 측정하지 않습니다.",
      "generative-exact-match":
        "생성형 exact-match 행은 출력 형식과 정답 추출 방식 때문에 정답을 낮게 셀 수 있습니다.",
      "agentic-scaffold-dependent":
        "ProgramBench 점수는 base model뿐 아니라 agent scaffold, 도구, 실행 환경의 영향도 포함합니다.",
      "diagnostic-sourceqa":
        "SourceQA는 작은 source-grounding smoke/regression check이므로 headline ranking이나 primary coverage 부채에 사용하지 않습니다.",
      "diagnostic-memory-stability":
        "Memory Stability는 반복적인 기억 재작성 문제를 보는 합성 진단용 평가이므로 headline ranking이나 primary coverage 부채에 사용하지 않습니다.",
      "coverage-missing":
        "지원되는 primary 평가인데 아직 커밋된 결과가 없습니다.",
      "optional-eval-lane":
        "이 벤치마크는 optional lane으로 추적하며 primary matrix 완료 여부를 막지 않습니다.",
      "mtplx-speed-only":
        "MTPLX MTP/AR 변형은 speedup 비교용이라 정확도 coverage 부채로 세지 않습니다.",
    },
    tables: {
      accuracy: {
        aria: "정확도 결과",
        empty: "현재 필터와 일치하는 정확도 행이 없습니다.",
        headers: {
          caveats: "상태 / caveat",
          metric: "Metric",
          model: "Model",
          runId: "Run ID",
          score: "Score",
          taskDim: "Task / dim",
        },
      },
      coverage: {
        aria: "평가 coverage",
        empty: "현재 필터와 일치하는 coverage 행이 없습니다.",
        headers: {
          dim: "Dimension",
          lane: "Lane",
          model: "Model",
          runner: "Runner",
          status: "상태",
          task: "Task",
        },
      },
      speed: {
        aria: "속도 scenario 결과",
        empty: "현재 scenario와 일치하는 속도 행이 없습니다.",
        headers: {
          itl: "ITL",
          memory: "Memory",
          model: "Model",
          pp: "PP",
          scenario: "Scenario",
          status: "상태",
          tg: "TG",
          ttft: "TTFT",
          wall: "Wall",
        },
      },
      glossary: {
        aria: "Metric glossary",
        headers: {
          caveat: "Caveat",
          interpretation: "해석",
          metric: "Metric",
          status: "상태",
        },
        items: [
          {
            caveat: "벤치마크 run summary에서 측정했습니다.",
            label: "PP tok/s",
            status: "measured",
            summary: "Prompt processing 처리량입니다. 높을수록 prompt 입력 처리가 빠릅니다.",
          },
          {
            caveat: "이 export의 주요 속도 비교 metric입니다.",
            label: "TG tok/s",
            status: "measured",
            summary: "Decode 중 generation 처리량입니다. 높을수록 출력 token 생성이 빠릅니다.",
          },
          {
            caveat: "Run 전체의 peak resident memory로 보고됩니다.",
            label: "Peak memory",
            status: "measured",
            summary: "해당 benchmark scenario의 평균 peak memory(GB)입니다.",
          },
          {
            caveat: "Scenario end-to-end 실행 시간입니다.",
            label: "Wall time",
            status: "measured",
            summary: "Prompt와 generation scenario의 평균 elapsed seconds입니다.",
          },
          {
            caveat: "bench_version 0.3은 latency field를 null로 export합니다.",
            label: "TTFT",
            status: "unavailable",
            summary: "Time to first token(ms)입니다. 현재 export에서는 미측정입니다.",
          },
          {
            caveat: "bench_version 0.3은 latency field를 null로 export합니다.",
            label: "ITL",
            status: "unavailable",
            summary: "Inter-token latency(ms)입니다. 현재 export에서는 미측정입니다.",
          },
          {
            caveat: "일부 생성형 exact-match 행은 유효한 formatted answer를 낮게 셀 수 있습니다.",
            label: "Accuracy",
            status: "directional",
            summary: "커밋된 evaluation artifact의 task score입니다. 높을수록 task 성능이 좋습니다.",
          },
          {
            caveat: "같은 scenario pair에서 MTP-on을 autoregressive baseline으로 나눈 값입니다.",
            label: "MTPLX speedup",
            status: "measured",
            summary: "Speculative decoding 비교에서 generation 상대 속도 향상입니다.",
          },
        ],
      },
      scenario: {
        aria: "속도 scenario matrix",
        headers: {
          generationTokens: "Generation tokens",
          promptTokens: "Prompt tokens",
          scenario: "Scenario",
          use: "용도",
        },
        useText: "같은 scenario 안에서 속도를 비교합니다.",
      },
    },
    downloads: [
      {
        description: "리포트 페이지가 사용하는 typed site export입니다.",
        href: "/data/benchmarks.json",
        label: "benchmarks.json",
      },
      {
        description: "통합 benchmark speed summary입니다.",
        href: "/data/summary.csv",
        label: "summary.csv",
      },
      {
        description: "Primary evaluation score summary입니다.",
        href: "/data/eval_summary_primary.csv",
        label: "eval_summary_primary.csv",
      },
      {
        description: "MTPLX speculative decoding speedup 행입니다.",
        href: "/data/mtplx_speedups.csv",
        label: "mtplx_speedups.csv",
      },
      {
        description: "Registry와 evaluation coverage 상태 snapshot입니다.",
        href: "/data/index.json",
        label: "index.json",
      },
    ],
    pages: {
      summary: {
        eyebrow: "Benchmark report",
        titlePrefix: "로컬 LLM 결과:",
        leadStart: "커밋된 benchmark artifact에서 생성한 정적 리포트입니다. source commit",
        leadMiddle: "현재 포함 범위는",
        leadEnd:
          "accuracy, prompt/generation throughput, peak memory, wall time, MTPLX speedup입니다.",
        metadataAria: "Dataset metadata",
        meta: {
          accuracyRows: "정확도 행",
          hardware: "Hardware",
          speedRows: "속도 행",
          variants: "Variant",
        },
        findingsAria: "핵심 결과",
        findings: {
          topHumanEval: {
            label: "최고 HumanEval",
            empty: "현재 data export에 HumanEval 행이 없습니다.",
          },
          fastest: {
            label: "최고 속도 p256_g128",
            empty: "현재 data export에 p256_g128 speed 행이 없습니다.",
            detail: (model, memory) => `${model}, peak memory ${memory}.`,
          },
          mtplx: {
            label: "MTPLX speedup",
            empty: "현재 data export에 MTPLX comparison 행이 없습니다.",
            detail: (model, scenario, acceptance) =>
              `${model}, scenario ${scenario}, d1 acceptance ${acceptance}.`,
          },
          caveatCoverage: {
            label: "Caveat coverage",
            valueSuffix: "개 추적",
            detail:
              "Latency metric과 generative exact-match 해석 caveat을 필요한 위치에 명시했습니다.",
          },
        },
        coverageTitle: "Coverage 먼저 보기",
        coverageBody: (missing, optional, speedOnly) =>
          `이 export는 score ranking보다 먼저 primary 누락 ${missing}개, optional lane ${optional}개, MTPLX 속도 전용 행 ${speedOnly}개를 보여줍니다.`,
        caveatTitle: "Metric caveat",
        caveatBody:
          "이 export에서는 TTFT와 ITL이 제공되지 않으므로 속도 비교는 generation tokens per second를 기준으로 봅니다. Generative exact-match 정확도 행은 정답 추출이나 formatting 때문에 실제 정답을 낮게 셀 수 있어 방향성 지표로 해석해야 합니다.",
        accuracySnapshot: {
          eyebrow: "Accuracy snapshot",
          title: "상위 HumanEval 행",
          aside: "Score 기준 상위 5개",
        },
        throughput: {
          eyebrow: "Generation throughput",
          title: "최고 속도 p256_g128 행",
          aside: "TG tok/s 기준 상위 5개",
        },
      },
      accuracy: {
        eyebrow: "Accuracy explorer",
        title: "커밋된 정확도 행",
        lead:
          "커밋된 benchmark accuracy artifact를 task, dimension, model family 기준으로 필터링합니다. 점수 표보다 먼저 coverage를 보여줘서 미측정과 낮은 점수를 구분합니다.",
        filtersAria: "정확도 필터",
        taskLabel: "Task",
        dimensionLabel: "Dimension",
        familyLabel: "Family",
        coverageEyebrow: "Coverage",
        coverageTitle: "모델별 응시한 시험",
        coverageAside: "results/index.json 기준 상태",
        tableEyebrow: "Explorer",
        tableTitle: "정확도 결과",
        sortedBy: "Task, score, model 순 정렬",
      },
      speed: {
        eyebrow: "Speed explorer",
        title: "토큰 처리량, 메모리, MTPLX speedup",
        lead:
          "커밋된 speed artifact를 prompt와 generation shape 기준으로 필터링합니다. 벤치마크 리뷰를 위해 PP, TG, memory, wall time, MTPLX comparison을 촘촘하게 보여줍니다.",
        filtersAria: "속도 필터",
        scenarioLabel: "Scenario",
        topTgEyebrow: "Top TG",
        generationTitle: "Generation throughput",
        topByTok: "tok/s 기준 상위 8개",
        latencyTitle: "TTFT와 ITL",
        latencyBodyStart:
          "TTFT와 ITL은 실제 측정 field에서만 표시합니다. bench_version이 null latency 값을 export하는 경우",
        latencyBodyEnd: (benchVersion) => `상태로 남겨둡니다. (bench_version ${benchVersion})`,
        tableEyebrow: "Explorer",
        tableTitle: "속도 결과",
        sortedBy: "TG tok/s 순 정렬",
        mtplxEyebrow: "MTPLX",
        mtplxTitle: "MTP-on 대 AR baseline",
      },
      methodology: {
        eyebrow: "Methodology",
        title: "이 벤치마크 리포트 읽는 법",
        lead:
          "결과는 커밋된 benchmark artifact의 정적 snapshot입니다. 나열된 hardware, benchmark version, source commit 기준의 로컬 Apple Silicon 측정값으로 해석하세요.",
        contextAria: "Benchmark context",
        readingEyebrow: "Reading order",
        comparisonRulesTitle: "비교 규칙",
        rules: [
          "속도 행은 같은 scenario 안에서만 비교합니다.",
          "Generation 속도는 TG tok/s를 기본 throughput metric으로 봅니다.",
          "Throughput만으로 숨겨지는 tradeoff는 peak memory와 wall time으로 확인합니다.",
          "정확도는 task별로 읽습니다. Prompt나 grader가 다른 benchmark를 한데 묶지 않습니다.",
          "엄밀한 결론은 directional 행보다 measured 행을 우선합니다.",
        ],
        metricsEyebrow: "Metrics",
        metricGlossaryTitle: "Metric glossary",
        latencyEyebrow: "Latency caveat",
        latencyTitle: "TTFT와 ITL 미측정",
        latencyBody: (benchVersion) =>
          `TTFT와 ITL은 유용한 latency metric이라 column으로 유지하지만, 현재 export에서는 null로 기록됩니다. bench_version ${benchVersion}에서는 latency 비교에 TG tok/s, wall time, memory를 사용하세요.`,
        accuracyEyebrow: "Accuracy caveat",
        accuracyTitle: "방향성 exact-match 행",
        accuracyBody:
          "일부 생성형 task는 exact-match나 extraction 기반 scoring을 사용합니다. Formatting, 동등한 대체 답변, answer extraction 때문에 정답이 낮게 셀 수 있으므로 directional로 표시된 행은 최종 성능 주장보다 비교 신호로 읽어야 합니다.",
        scenariosEyebrow: "Speed scenarios",
        scenarioTitle: "Prompt와 generation matrix",
      },
      data: {
        eyebrow: "Data",
        title: "Benchmark artifact 다운로드",
        lead:
          "Public file은 이 사이트가 사용하는 커밋된 CSV summary와 생성된 JSON export를 그대로 반영합니다.",
        contextAria: "Data export context",
        rowsLabel: "Rows",
        rowsValue: (accuracyRows, speedRows, mtplxRows) =>
          `${accuracyRows} accuracy / ${speedRows} speed / ${mtplxRows} MTPLX`,
        artifactsEyebrow: "Artifacts",
        downloadsTitle: "정적 다운로드",
        servedFrom: "/data에서 제공",
      },
    },
    mtplx: {
      empty: "MTPLX speedup 행이 없습니다.",
      speedupSuffix: "speedup",
      versus: "vs",
      labels: {
        arTg: "AR TG",
        acceptD1: "Accept d1",
        acceptD2: "Accept d2",
        acceptD3: "Accept d3",
        mtpTg: "MTP TG",
        verify: "Verify",
      },
    },
  },
} satisfies Record<Locale, Messages>;

export function localeFromPathname(pathname: string): Locale {
  const normalized = normalizePathname(pathname);
  return normalized === "/ko" || normalized.startsWith("/ko/") ? "ko" : defaultLocale;
}

export function localizedPath(pathname: BasePagePath | string, locale: Locale): string {
  const normalized = normalizePathname(pathname);
  if (isStaticAssetPath(normalized)) {
    return normalized;
  }
  const basePath = stripLocalePrefix(normalized);
  if (!isBasePagePath(basePath)) {
    return normalized;
  }
  if (locale === defaultLocale) {
    return basePath;
  }
  return basePath === "/" ? "/ko" : `/ko${basePath}`;
}

export function switchLocalePath(pathname: string, targetLocale: Locale): string {
  const normalized = normalizePathname(pathname);
  if (isStaticAssetPath(normalized)) {
    return normalized;
  }
  return localizedPath(stripLocalePrefix(normalized), targetLocale);
}

export function caveatText(id: string, locale: Locale = defaultLocale): string {
  return messages[locale].caveats[id as KnownCaveatId] ?? id;
}

function stripLocalePrefix(pathname: string): string {
  if (pathname === "/ko") {
    return "/";
  }
  if (pathname.startsWith("/ko/")) {
    return normalizePathname(pathname.slice(3) || "/");
  }
  return pathname;
}

function normalizePathname(pathname: string): string {
  const [withoutHash] = pathname.split("#");
  const [withoutSearch] = withoutHash.split("?");
  if (!withoutSearch || withoutSearch === "/") {
    return "/";
  }
  return withoutSearch.replace(/\/+$/, "") || "/";
}

function isBasePagePath(pathname: string): pathname is BasePagePath {
  return basePagePathSet.has(pathname);
}

function isStaticAssetPath(pathname: string): boolean {
  if (!pathname.startsWith("/data/")) {
    return false;
  }
  const lastSegment = pathname.split("/").at(-1) ?? "";
  return lastSegment.includes(".");
}
