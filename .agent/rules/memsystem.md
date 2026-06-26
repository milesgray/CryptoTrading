---
trigger: always_on
---

REMEMBER: After every memory reset, I begin completely fresh. The Memory Bank is my only link to previous work. It must be maintained with precision and clarity, as my effectiveness depends entirely on its accuracy.

## Workflow Diagrams

### Initialization Workflow
```mermaid
flowchart TD
    Start[Start] --> checkMemoryBankExists{checkMemoryBankExists}

    checkMemoryBankExists -->|No| createMemoryBankDirectory[createMemoryBankDirectory]
    createMemoryBankDirectory --> scaffoldMemoryBankStructure[scaffoldMemoryBankStructure]
    scaffoldMemoryBankStructure --> populateMemoryBankFiles[populateMemoryBankFiles]
    populateMemoryBankFiles --> readMemoryBank[readMemoryBank]

    checkMemoryBankExists -->|Yes| readMemoryBank

    readMemoryBank --> verifyFilesComplete{verifyFilesComplete}

    verifyFilesComplete -->|No| createMissingFiles[createMissingFiles]
    createMissingFiles --> verifyContext[verifyContext]

    verifyFilesComplete -->|Yes| verifyContext

    verifyContext --> developStrategy[developStrategy]
```

### Documentation Workflow
```mermaid
flowchart TD
    Start[Start] --> checkDocumentationExists{checkDocumentationExists}
    checkDocumentationExists -->|No| scaffoldDocumentationStructure[scaffoldDocumentationStructure]
    checkDocumentationExists -->|Yes| generateDocumentation[generateDocumentation]
    scaffoldDocumentationStructure --> generateDocumentation
    generateDocumentation --> selfEvaluateDocumentation[selfEvaluateDocumentation]
    selfEvaluateDocumentation --> reviewDocumentation[reviewDocumentation]
    reviewDocumentation -->|Score < 4| reviseDocumentation[reviseDocumentation]
    reviewDocumentation -->|Score >= 4| updateMemoryBank[updateMemoryBank]
    reviseDocumentation -->|Improved| updateMemoryBank
    reviseDocumentation -->|Still Failing| rejectAndFlag[rejectAndFlag]
    updateMemoryBank --> calculateDocumentationQualityScore[calculateDocumentationQualityScore]
```

### Implementation Workflow
```mermaid
flowchart TD
    Start[Start] --> executeTask[executeTask]
    executeTask --> checkMemoryBank[checkMemoryBank]
    checkMemoryBank --> updateDocumentation[updateDocumentation]
    updateDocumentation --> updatePlans[updatePlans]
    updatePlans --> executeImplementation[executeImplementation]
    executeImplementation --> enforceCodeQualityStandards[enforceCodeQualityStandards]
    enforceCodeQualityStandards --> executeCreatorPhase[executeCreatorPhase]
    executeCreatorPhase --> executeCriticPhase[executeCriticPhase]
    executeCriticPhase --> executeDefenderPhase[executeDefenderPhase]
    executeDefenderPhase --> executeJudgePhase[executeJudgePhase]
```

### Error Recovery Workflow
```mermaid
flowchart TD
    Start[Start] --> detectToolFailure[detectToolFailure]
    detectToolFailure --> logFailureDetails[logFailureDetails]
    logFailureDetails --> analyzeFailureCauses[analyzeFailureCauses]
    analyzeFailureCauses --> reviewToolUsage[reviewToolUsage]
    reviewToolUsage --> adjustParameters[adjustParameters]
    adjustParameters --> executeRetry[executeRetry]
    executeRetry --> checkRetrySuccess{checkRetrySuccess}

    checkRetrySuccess -->|Success| continueTask[Continue Task]
    checkRetrySuccess -->|Failure| incrementRetryCount[incrementRetryCount]
    incrementRetryCount --> checkRetryLimit{checkRetryLimit}

    checkRetryLimit -->|Under Limit| executeRetry
    checkRetryLimit -->|Limit Reached| escalateToUser[escalateToUser]
    escalateToUser --> documentFailure[documentFailure]
    documentFailure --> alertUser[alertUser]
```

### Evaluation Workflow
```mermaid
flowchart TD
    Start[Start] --> documentObjectiveSummary[documentObjectiveSummary]
    documentObjectiveSummary --> calculatePerformanceScore[calculatePerformanceScore]
    calculatePerformanceScore --> evaluateAgainstTargetScore[evaluateAgainstTargetScore]
    evaluateAgainstTargetScore --> checkPerformance{performanceScore < targetScore}

    checkPerformance -->|Yes| analyzePerformanceGap[analyzePerformanceGap]
    analyzePerformanceGap --> identifyImprovementOpportunities[identifyImprovementOpportunities]
    identifyImprovementOpportunities --> implementOptimizations[implementOptimizations]
    implementOptimizations --> recalculatePerformanceScore[recalculatePerformanceScore]
    recalculatePerformanceScore --> checkTargetAchieved{checkTargetAchieved}

    checkPerformance -->|No| recordSuccessPatterns[recordSuccessPatterns]

    checkTargetAchieved -->|Yes| recordSuccessPatterns
    checkTargetAchieved -->|No| iterateOptimizationCycle[iterateOptimizationCycle]
    iterateOptimizationCycle --> implementOptimizations

    recordSuccessPatterns --> documentLessonsLearned[documentLessonsLearned]
    documentLessonsLearned --> updateMemoryBank[updateMemoryBank]
```

### Self-Critique Workflow
```mermaid
flowchart TD
    Start[Start] --> executeCreatorPhase[executeCreatorPhase]
    executeCreatorPhase --> executeCriticPhase[executeCriticPhase]
    executeCriticPhase --> executeDefenderPhase[executeDefenderPhase]
    executeDefenderPhase --> executeJudgePhase[executeJudgePhase]
``

## Event Handlers

<EventHandlers>
  <Handler event="SessionStart">
    <Action>Check if `.agent/` directory structure exists</Action>
    <Action>If structure doesn't exist, scaffold it by creating all required directories</Action>
    <Action>If memory files don't exist, initialize them with available project information</Action>
    <Action>Load all memory layers from `.agent/core/`</Action>
    <Action>Verify memory consistency using checksums in memory-index.md</Action>
    <Action>Identify current task context from activeContext.md</Action>
    <Action>Create a memory of this initialization process using the CASCADE GENERATED MEMORY system for automatic reminder via EPHEMERAL MEMORY</Action>
  </Handler>

  <Handler event="TaskStart">
    <Action>Document task objectives in new task log</Action>
    <Action>Develop criteria for successful task completion</Action>
    <Action>Load relevant context from memory</Action>
    <Action>Create implementation plan</Action>
  </Handler>

  <Handler event="ErrorDetected">
    <Action>Document error details in `.agent/errors/`</Action>
    <Action>Check memory for similar errors</Action>
    <Action>Apply recovery strategy</Action>
    <Action>Update error patterns</Action>
  </Handler>

  <Handler event="TaskComplete">
    <Action>Document implementation details in task log</Action>
    <Action>Evaluate performance</Action>
    <Action>Update all memory layers</Action>
    <Action>Update activeContext.md with next steps</Action>
  </Handler>

  <Handler event="SessionEnd">
    <Action>Ensure all memory layers are synchronized</Action>
    <Action>Document session summary in activeContext.md</Action>
    <Action>Update checksums in memory-index.md</Action>
  </Handler>
</EventHandlers>
