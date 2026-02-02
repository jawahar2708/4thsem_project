"""mermaid
useCaseDiagram
    actor "Patient" as P
    actor "Clinician / Psychologist" as C
    package "Psychopathology Analysis System" {
        usecase "Record Session" as UC1
        usecase "Upload Video" as UC2
        usecase "Analyze Visual Cues\n(FER / Micro-expressions)" as UC3
        usecase "Analyze Audio Tone\n(Prosody / Pitch)" as UC4
        usecase "Generate Report" as UC5
        usecase "View Historical Trend" as UC6
    }
    %% Associations
    P -- UC1 : Participates in
    C -- UC1 : Initiates
    C -- UC2 : Uploads input
    C -- UC5 : Requests
    C -- UC6 : Monitors progress
    %% Relationships (Includes/Extends)
    UC2 ..> UC3 : <<include>>
    UC2 ..> UC4 : <<include>>
"""
    UC5 ..> UC3 : <<include>>
    UC5 ..> UC4 : <<include>>
